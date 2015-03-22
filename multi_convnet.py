import cPickle
import os
import sys
import time
from collections import OrderedDict
import numpy

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from logistic_sgd import LogisticRegression, shared_dataset
from cifar10 import CIFAR10
from cifar100 import CIFAR100
from mnist import MNIST
from svhn import SVHN
from norb import NORB
from pylearn2.datasets import preprocessing
from mlp import HiddenLayer,ReLU,Sigmoid,Tanh,DropoutHiddenLayer,_dropout_from_layer
from convolutional_mlp import LeNetConvPoolLayer

def stochastic_select(rng,a,b):
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    mask = srng.random_integers(size=a.shape)    
    output = T.switch(mask,a,b)
    return(output)
#from show_filters import show_filters
class Convnet_Params(object):
    def __init__(self,numpy_rng,conv_activations,no_pools,
                 pooltypes,strides,poolsizes,
                 kernsizes,nkerns,filter_pads,no_norms):
        self.numpy_rng = numpy_rng
        self.conv_activations = conv_activations
        self.no_pools =  no_pools
        self.pooltypes = pooltypes
        self.strides = strides
        self.poolsizes = poolsizes
        self.kernsizes = kernsizes
        self.nkerns = nkerns
        self.filter_pads=filter_pads
        self.no_norms=no_norms
class Convnet(object):
    

    def __init__(self, numpy_rng, input,n_ins,
                 batch_size,channel,conv_activations,
                 no_pools,pooltypes,strides,filter_pads,
                 poolsizes,kernsizes,nkerns,no_norms):
                     
        self.params = []
        n_kerns = len(nkerns)
        assert n_kerns > 0
        convpool_layer=None
        image_size=None
        for i in xrange(n_kerns):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer

            if i == 0:
                filter_shape = (nkerns[i], channel, kernsizes[i], kernsizes[i])
            else:
                filter_shape = (nkerns[i], nkerns[i-1], kernsizes[i],kernsizes[i])

            if i == 0:
                image_shape = (batch_size, channel, n_ins[0]+filter_pads[i]*2, 
                               n_ins[1]+filter_pads[i]*2)
            else:
                if no_pools[i-1]:
                    image_size =  image_shape[2]+filter_pads[i]*2-kernsizes[i-1]+1
                else:
                    image_size = image_shape[2]+filter_pads[i]*2-kernsizes[i-1]+1
                    image_size = int(numpy.ceil(float(image_size-poolsizes[i][0]) / strides[i])) + 1                   
                image_shape = (batch_size, nkerns[i - 1], image_size, image_size)
            #print image_size
            if i == 0:
                layer_input = input.dimshuffle(1, 2, 3, 0)
            else:
                layer_input = convpool_layer.output

            convpool_layer = LeNetConvPoolLayer(numpy_rng, input=layer_input,
                             image_shape=image_shape,
                             filter_shape=filter_shape,
                             no_pool = no_pools[i],
                             pooltype=pooltypes[i],
                             poolsize=poolsizes[i],
                             conv_act=conv_activations[i],
                             stride = strides[i],
                             filter_pad=filter_pads[i],
                             no_norm=no_norms[i])
            
            self.params.extend(convpool_layer.params)
            if i==0:
                self.L1 = abs(convpool_layer.W).sum()
                self.L2_sqr = (convpool_layer.W ** 2).sum()
            else:
                self.L1 += abs(convpool_layer.W).sum()
                self.L2_sqr += (convpool_layer.W ** 2).sum()
            
        if no_pools[i]:
            image_size =  image_shape[2]+filter_pads[-1]*2-kernsizes[-1]+1
        else:
            image_size = image_shape[2]+filter_pads[-1]*2-kernsizes[-1]+1
            image_size = int(numpy.ceil(float(image_size - poolsizes[-1][0]) / strides[-1]))#+1
            print image_size
        #if kernsizes[0]==5:
            #image_size = image_size-1
        
        self.out_layer_size = nkerns[-1]*image_size*image_size
        self.layer_out = convpool_layer.output.dimshuffle(3, 0, 1, 2).flatten(2)
class MultiConvnet(object):
    def __init__(self, numpy_rng, n_ins,
                 batch_size,channel,convnet_params,
                 convnet_weights,combine_type,
                 layer_activations,dropout_rates,  
                 layer_sizes):
        
        self.batch_size = batch_size
        self.params =[]
      
        self.x = T.matrix('x')  
        self.y = T.ivector('y')
        
        convnet_input = self.x.reshape((self.batch_size, channel, n_ins[0], n_ins[1]))       
        convnet_count = 0
        layer_input=None
        for convnet_param in convnet_params:
            convnet = Convnet(numpy_rng=convnet_param.numpy_rng,
                              input=convnet_input,n_ins=n_ins,
                              batch_size=self.batch_size,
                              channel=channel,
                              no_pools=convnet_param.no_pools,
                              pooltypes=convnet_param.pooltypes,
                              poolsizes=convnet_param.poolsizes,
                              strides=convnet_param.strides,
                              kernsizes=convnet_param.kernsizes,
                              conv_activations=convnet_param.conv_activations,
                              nkerns=convnet_param.nkerns,
                              filter_pads=convnet_param.filter_pads,
                              no_norms=convnet_param.no_norms)
            self.params.extend(convnet.params)
            if convnet_count == 0:
                if combine_type=='average':
                    layer_input =convnet_weights[convnet_count]*convnet.layer_out
                elif combine_type=='max':
                    layer_input =convnet.layer_out
                elif combine_type=='min':
                    layer_input =convnet.layer_out
                self.L1 = convnet.L1
                self.L2_sqr = convnet.L2_sqr
            else:
                if combine_type=='average':
                    layer_input +=convnet_weights[convnet_count]*convnet.layer_out
                elif combine_type=='max':
                    layer_input =T.maximum(layer_input,convnet.layer_out)
                elif combine_type=='min':
                    layer_input =T.minimum(layer_input,convnet.layer_out)#stochastic_select(numpy_rng,layer_input,convnet.layer_out)
                self.L1 += convnet.L1
                self.L2_sqr += convnet.L2_sqr
            convnet_count += 1
            
        out_layer_size = convnet.out_layer_size  
        layer_sizes.insert(0,out_layer_size)
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        next_layer_input = layer_input
        #next_layer_input = layer_input*(1 - dropout_rates[0])
        next_dropout_layer_input = _dropout_from_layer(numpy_rng, next_layer_input, p=dropout_rates[0])
        #next_dropout_layer_input = next_layer_input
        layer_counter = 0        
        for n_in, n_out in weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=numpy_rng,
                    input=next_dropout_layer_input,
                    activation=layer_activations[layer_counter],
                    n_in=n_in, n_out=n_out,
                    dropout_rate=dropout_rates[layer_counter])
            next_dropout_layer_input = next_dropout_layer.output
            
            self.params.extend(next_dropout_layer.params)
            self.L1 += abs(next_dropout_layer.W).sum()
            self.L2_sqr += (next_dropout_layer.W ** 2).sum()
            
            next_layer = HiddenLayer(
                    input=next_layer_input,
                    rng=numpy_rng,
                    activation=layer_activations[layer_counter],
                    # scale the weight matrix W with (1-p)
                    W=next_dropout_layer.W* (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out)

            next_layer_input = next_layer.output
            layer_counter += 1
           
        n_in, n_out = weight_matrix_sizes[-1]
        dropout_classifier_layer  = LogisticRegression(
                input=next_dropout_layer_input,
                rng=numpy_rng,
                n_in=n_in, n_out=n_out)
        self.L1 += abs(dropout_classifier_layer.W).sum()
        self.L2_sqr += (dropout_classifier_layer.W ** 2).sum()
        
        classifier_layer = LogisticRegression(               
                input=next_layer_input,
                rng=numpy_rng,
                # scale the weight matrix W with (1-p)
                W=dropout_classifier_layer.W *0.5,#* (1 - dropout_rates[layer_counter])
                b=dropout_classifier_layer.b,
                n_in=n_in, n_out=n_out)

        self.dropout_cost = dropout_classifier_layer.negative_log_likelihood(self.y)
        self.dropout_errors = dropout_classifier_layer.errors(self.y)

        self.cost = classifier_layer.negative_log_likelihood(self.y)
        self.errors = classifier_layer.errors(self.y)
        self.params.extend(dropout_classifier_layer.params)
    def build_finetune_functions(self, datasets,
                            initial_learning_rate,
                            weight_decay,
                            learning_rate_decay,
                            squared_filter_length_limit,
                            mom_params,dropout,L1_reg,L2_reg):
        learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
            dtype=theano.config.floatX))

        mom_start = mom_params["start"]
        mom_end = mom_params["end"]
        mom_epoch_interval = mom_params["interval"]
        
        (train_set_x, train_set_y) = shared_dataset(datasets[0].get_data())
        (test_set_x, test_set_y) = shared_dataset(datasets[1].get_data())
        #(train_set_x, train_set_y) = datasets[0]
        #(test_set_x, test_set_y) = datasets[1]

        # compute number of minibatches for training, validation and testing
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= self.batch_size

        index = T.lscalar('index')  # index to a [mini]batch   
        epoch = T.scalar()
        # compute the gradients with respect to the model parameters
        gparams = []
        cost = self.dropout_cost if dropout else self.cost
        cost += L1_reg * self.L1 + L2_reg * self.L2_sqr
        for param in self.params:
            # Use the right cost function here to train with or without dropout.
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        # ... and allocate mmeory for momentum'd versions of the gradient
        gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                dtype=theano.config.floatX))
            gparams_mom.append(gparam_mom)

        # Compute momentum for the current epoch
        #mom = ifelse(epoch < mom_epoch_interval,
                #mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end*(epoch/mom_epoch_interval),
                #mom_end)

        # Update the step direction using momentum
        updates = OrderedDict()
        for param,gparam_mom, gparam in zip(self.params,gparams_mom, gparams):
            # Misha Denil's original version
            #updates[gparam_mom] = mom * gparam_mom + (1. - mom) * gparam
      
            # change the update rule to match Hinton's dropout paper
            #updates[gparam_mom] = mom * gparam_mom - (1. - mom) * learning_rate * gparam
            updates[gparam_mom] = (mom_start * gparam_mom - learning_rate * gparam-
                                  weight_decay*learning_rate*param)

        # ... and take a step along that direction
        for param, gparam_mom in zip(self.params, gparams_mom):
            # Misha Denil's original version
            #stepped_param = param - learning_rate * updates[gparam_mom]
            
            # since we have included learning_rate in gparam_mom, we don't need it
            # here
            stepped_param = param + updates[gparam_mom]

            # This is a silly hack to constrain the norms of the rows of the weight
            # matrices.  This just checks if there are two dimensions to the
            # parameter and constrains it if so... maybe this is a bit silly but it
            # should work for now.
            if param.get_value(borrow=True).ndim == 2:
                #squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
                #scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
                #updates[param] = stepped_param * scale
                
                # constrain the norms of the COLUMNs of the weight, according to
                # https://github.com/BVLC/caffe/issues/109
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(squared_filter_length_limit))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param


        # Compile theano function for training.  This returns the training cost and
        # updates the model parameters.
        output = cost
        train_fn = theano.function(inputs=[index], outputs=output,
                updates=updates,
                givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]},
                name='train')
        #theano.printing.pydotprint(train_model, outfile="train_file.png",
        #        var_with_name_simple=True)

        # Theano function to decay the learning rate, this is separate from the
        # training function because we only want to do this once each epoch instead
        # of after each minibatch.
        decay_learning_rate = theano.function(inputs=[epoch], outputs=learning_rate,
                updates={learning_rate: ifelse(epoch%20,learning_rate,learning_rate/10)})

        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * self.batch_size:
                                      (index + 1) * self.batch_size],
                   self.y: test_set_y[index * self.batch_size:
                                      (index + 1) * self.batch_size]},
                      name='test')

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, test_score,decay_learning_rate

def test_MulitConvnet(n_ins,batch_size,channel,convnet_params,
                 layer_activations,dropout_rates,  
                 layer_sizes,convnet_weights,
                 combine_type,L1_reg,L2_reg,
                 initial_learning_rate,
                 weight_decay,data_type,
                 learning_rate_decay,random_seed,
                 squared_filter_length_limit,
                 mom_params,dropout,output_folder,
                 results_file,training_epochs):
    
    pipeline = preprocessing.Pipeline()
    if data_type=='CIFAR10':
        trainsets = CIFAR10(which_set='train',gcn=55)
        testsets = CIFAR10(which_set='test',gcn=55)
        pipeline.items.append(preprocessing.ZCA())
        trainsets.apply_preprocessor(preprocessor = pipeline, can_fit = True)
        testsets.apply_preprocessor(preprocessor = pipeline, can_fit = False)
    elif data_type=='CIFAR100':
        trainsets = CIFAR100(which_set='train',gcn=55)
        testsets = CIFAR100(which_set='test',gcn=55)
        pipeline.items.append(preprocessing.ZCA())
        trainsets.apply_preprocessor(preprocessor = pipeline, can_fit = True)
        testsets.apply_preprocessor(preprocessor = pipeline, can_fit = False)
    elif data_type=='SVHN':
        trainsets = SVHN(which_set='splitted_train')#
        testsets = SVHN(which_set='test')
        pipeline.items.append(preprocessing.GlobalContrastNormalization(batch_size=5000))
        pipeline.items.append(preprocessing.LeCunLCN((32,32)))
        trainsets.apply_preprocessor(preprocessor = pipeline, can_fit = True)
        testsets.apply_preprocessor(preprocessor = pipeline, can_fit = False)
    elif data_type=='MNIST':
        trainsets = MNIST(which_set='train')
        testsets = MNIST(which_set='test')
        #pipeline.items.append(preprocessing.ZCA())
    elif data_type=='NORB':
        trainsets = NORB(which_set='train',gcn=55)
        testsets = NORB(which_set='test',gcn=55)
    
    datasets=[trainsets,testsets]
    train_set_x, train_set_y = trainsets.get_data()
    #datasets = load_data(dataset)

    #train_set_x, train_set_y = datasets[0]
    # compute number of minibatches for training, validation and testing
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = train_set_x.shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    rng = numpy.random.RandomState(random_seed)
    #theano_rng = RandomStreams(rng.randint(2 ** 30))
    print '... building the model'
    # construct the stacked denoising autoencoder class
    multiConvnet = MultiConvnet(numpy_rng=rng, n_ins=n_ins,
                                batch_size=batch_size,
                                channel=channel,
                                convnet_params=convnet_params,
                                layer_activations=layer_activations,
                                dropout_rates=dropout_rates,  
                                layer_sizes=layer_sizes,
                                convnet_weights=convnet_weights,
                                combine_type=combine_type)

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn,test_model,decay_learning_fn = multiConvnet.build_finetune_functions(
                datasets=datasets,
                initial_learning_rate=initial_learning_rate,
                weight_decay=weight_decay,
                learning_rate_decay=learning_rate_decay,
                squared_filter_length_limit=squared_filter_length_limit,
                mom_params=mom_params,L1_reg=L1_reg,L2_reg=L2_reg,
                dropout=dropout)

    print '... finetunning the model'

    best_params = None
    best_score = numpy.inf
    start_time = time.clock()

    done_looping = False
    epoch = 0
    best_epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            """print('epoch %i, minibatch %i/%i, minibatch_avg_cost %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       minibatch_avg_cost))"""

        test_losses = test_model()
        test_score = numpy.mean(test_losses)
        print(('     epoch %i, minibatch %i/%i, test error of '
               'best model %f %%') %
              (epoch, minibatch_index + 1, n_train_batches,
               test_score * 100.))
        results_file.write('%i %f\n'%(epoch,test_score * 100.))
        results_file.flush()
        if test_score < best_score:
            best_score=test_score
            best_epoch=epoch
        new_learning_rate = decay_learning_fn(epoch)
    end_time = time.clock()
    print(('Optimization complete with test performance %f in epoch %i %%') %
                 (best_score * 100.,best_epoch))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    results_file.write('%.2fm'%((end_time - start_time) / 60))
    results_file.flush()
    #show_filters(best_params,output_folder)
if __name__ == '__main__':
    
    ####################
    # Convnet Params  #
    ###################
    convnet_params=[]
    #########################
    # First Convnet Params  #
    ########################
    numpy_rng=numpy.random.RandomState(89677)#23455
    conv_activations=[ReLU,ReLU,ReLU]#ReLU,Sigmoid,Tanh
    no_pools=[False,False,False]
    pooltypes=['max_pool','max_pool','max_pool']#max_pool,average_pool,stochastic_pool
    strides = [2,2,2]
    poolsizes=[(4,4),(4,4),(4,4)]
    filter_pads=[2,2,2]
    kernsizes=[5,5,5]
    nkerns=[64,64,96]#[32,80,96] [64,64,96]
    no_norms=[False,False,True]
    convnet_param=Convnet_Params( numpy_rng=numpy_rng,
                                  conv_activations=conv_activations,
                                  no_pools=no_pools,pooltypes=pooltypes,
                                  strides=strides,poolsizes=poolsizes,
                                  kernsizes=kernsizes,nkerns=nkerns,
                                  filter_pads=filter_pads,no_norms=no_norms)
    convnet_params.append(convnet_param)
    ##########################
    # Second Convnet Params  #
    #########################
    
    numpy_rng=numpy.random.RandomState(23455)#23455
    conv_activations=[ReLU,ReLU,ReLU]#ReLU,Sigmoid,Tanh
    no_pools=[False,False,False]
    pooltypes=['max_pool','max_pool','max_pool']#max_pool,average_pool,stochastic_pool
    strides = [2,2,2]
    poolsizes=[(4,4),(4,4),(4,4)]
    filter_pads=[2,2,2]
    kernsizes=[5,5,5]
    nkerns=[64,64,96]#[32,80,96] [64,64,96]
    no_norms=[False,False,True]#"""
    convnet_param=Convnet_Params(numpy_rng=numpy_rng,
                                  conv_activations=conv_activations,
                                  no_pools=no_pools,pooltypes=pooltypes,
                                  strides=strides,poolsizes=poolsizes,
                                  kernsizes=kernsizes,nkerns=nkerns,
                                  filter_pads=filter_pads,no_norms=no_norms)
    convnet_params.append(convnet_param)
    
    ##########################
    # Thired Convnet Params  #
    #########################
    numpy_rng=numpy.random.RandomState(1234)
    conv_activations=[ReLU,ReLU,ReLU]#ReLU,Sigmoid,Tanh
    no_pools=[False,False,False]
    pooltypes=['max_pool','max_pool','max_pool']#max_pool,average_pool,stochastic_pool
    strides = [2,2,2]
    poolsizes=[(4,4),(4,4),(4,4)]#[(3,3),(3,3)]
    filter_pads=[2,2,2]#[2,1]
    kernsizes=[5,5,5]
    nkerns=[64,64,96]#[32,80,96] [64,64,96]
    no_norms=[False,False,True]
    convnet_param=Convnet_Params(numpy_rng=numpy_rng,
                                  conv_activations=conv_activations,
                                  no_pools=no_pools,pooltypes=pooltypes,
                                  strides=strides,poolsizes=poolsizes,
                                  kernsizes=kernsizes,nkerns=nkerns,
                                  filter_pads=filter_pads,no_norms=no_norms)
    convnet_params.append(convnet_param)
    
    """numpy_rng=numpy.random.RandomState(76429)
    conv_activations=[ReLU,ReLU,ReLU]#ReLU,Sigmoid,Tanh
    no_pools=[False,False,False]
    pooltypes=['max_pool','max_pool','max_pool']#max_pool,average_pool,stochastic_pool
    strides = [2,2,2]
    poolsizes=[(3,3),(3,3),(3,3)]
    filter_pads=[2,2,2]
    kernsizes=[5,5,5]
    nkerns=[64,64,96]#[32,80,96] [64,64,96]
    no_norms=[False,False,True]
    convnet_param=Convnet_Params(numpy_rng=numpy_rng,
                                  conv_activations=conv_activations,
                                  no_pools=no_pools,pooltypes=pooltypes,
                                  strides=strides,poolsizes=poolsizes,
                                  kernsizes=kernsizes,nkerns=nkerns,
                                  filter_pads=filter_pads,no_norms=no_norms)
    #convnet_params.append(convnet_param)"""
  
  
     #####################
    # Classifier Params  #
    #####################
    data_type = 'MNIST'#'CIFAR10','CIFAR100','MNIST','SVHN','NORB'
    n_ins = (32,32)    
    channel=3
    if data_type == 'MNIST':
        n_ins = (28,28)    
        channel=1
    elif data_type == 'NORB':
        n_ins = (48,48)    
        channel=2
    layer_activations = [ReLU]
    dropout_rates = [0.2,0.5]
    layer_sizes = [500,10]
    if data_type == 'CIFAR100':
        layer_sizes = [500,100]
    elif data_type == 'NORB':
        layer_sizes = [500,6]
    convnet_weights=[0.5,0.5]
    combine_type='max'  #'max','average','stochastic_select' ,'min'
    #################
    # Train Params  #
    #################    
    random_seed = 23455
    initial_learning_rate = 0.01
    learning_rate_decay = 0.993
    weight_decay=0.0005
    squared_filter_length_limit = 15.0
    batch_size = 128
    dropout = True
    L1_reg=0.00
    L2_reg=0.00#0.0001
    #results_file_name = "result/{0}.txt".format(data_type)
    results_file_name = "newresult/MNIST425[64,64,96].txt"#325323
    results_file = file(results_file_name,'w')
    
    #### the params for momentum
    mom_start = 0.9
    mom_end = 0.9
    # for epoch in [0, mom_epoch_interval], the momentum increases linearly
    # from mom_start to mom_end. After mom_epoch_interval, it stay at mom_end
    mom_epoch_interval = 250
    mom_params = {"start": mom_start,
                  "end": mom_end,
                  "interval": mom_epoch_interval}
                  
    output_folder = "filters_plots"
    training_epochs=250

    test_MulitConvnet(n_ins=n_ins,batch_size=batch_size,
                      channel=channel,convnet_params=convnet_params, 
                      layer_activations=layer_activations,
                      dropout_rates=dropout_rates,  
                      layer_sizes=layer_sizes,
                      convnet_weights=convnet_weights,
                      combine_type=combine_type,
                      L1_reg=L1_reg,L2_reg=L2_reg,
                      initial_learning_rate=initial_learning_rate,
                      learning_rate_decay=learning_rate_decay,
                      weight_decay=weight_decay,data_type=data_type,
                      random_seed=random_seed,
                      squared_filter_length_limit=squared_filter_length_limit,
                      mom_params=mom_params,dropout=dropout,
                      output_folder=output_folder,
                      results_file=results_file,training_epochs=training_epochs)       
