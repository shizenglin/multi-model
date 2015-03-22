"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import numpy

import theano
import theano.tensor.shared_randomstreams

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool
from pylearn2.sandbox.cuda_convnet.stochastic_pool import StochasticMaxPool
from pylearn2.expr.normalize import CudaConvNetCrossChannelNormalization


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape,
                 no_pool,conv_act,pooltype,image_shape,
                 poolsize,stride,filter_pad,no_norm,
                 alpha=0.001, beta=0.75, size_f=9, blocked=True):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        conv_op = FilterActs(pad=filter_pad, stride=1, partial_sum=1)
        contiguous_input = gpu_contiguous(input)
        contiguous_filters = gpu_contiguous(filters_shuffled)
        conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)
        conv_out_shuffled = conv_act(conv_out_shuffled + self.b.dimshuffle(0,'x', 'x', 'x'))
        pooled_out=None
        if(no_pool):
            pooled_out = conv_out_shuffled
        else:
            if pooltype=='max_pool':
                pool_op = MaxPool(ds=poolsize[0], stride=stride)
            elif pooltype=='average_pool':
                pool_op = MaxPool(ds=poolsize[0], stride=stride)
            elif pooltype=='stochastic_pool':
                pool_op = StochasticMaxPool(ds=poolsize[0], stride=stride)

            pooled_out_shuffled = pool_op(conv_out_shuffled)
             # c01b to bc01
            if no_norm:
                pooled_out = pooled_out_shuffled
            else:
                output_shuffled = gpu_contiguous(pooled_out_shuffled)
                response_norm=CudaConvNetCrossChannelNormalization(size_f=size_f, 
                                alpha=alpha,beta=beta, blocked=blocked)
                output_shuffled =response_norm(output_shuffled)
                pooled_out = output_shuffled
        
        self.output = pooled_out
        # downsample each feature map individually, using maxpoolin

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

        # store parameters of this layer
        self.params = [self.W, self.b]
