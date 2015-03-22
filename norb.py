import cPickle
import pickle
import numpy
class NORB():
    def __init__(self,which_set,gcn=None):
        data_path = '/home/shizenglin/masterwork/data/norb/'
        NUM_IMAGE_BATCH = 29160
        IMAGE_TARGET_SIZE = 48
        self.x = None
        self.y = None
        if which_set == 'train':
            train_set_x = numpy.zeros( (2*NUM_IMAGE_BATCH,2*(IMAGE_TARGET_SIZE)**2))
            dataset = data_path+'data_batch_1'
            f = open(dataset, 'rb')
            train_set= pickle.load(f)
            train_set_x[0:NUM_IMAGE_BATCH,:]=train_set['data']
            train_set_y=train_set['labels']
            f.close()
            dataset = data_path+'data_batch_2'
            f = open(dataset, 'rb')
            train_set= pickle.load(f)
            train_set_x[NUM_IMAGE_BATCH:2*NUM_IMAGE_BATCH,:]= train_set['data']
            train_set_y += train_set['labels']
            train_set_y = numpy.asarray(train_set_y).astype('float32')
            f.close()
            """train_set_x = train_set_x / 255.
            train_set_x = train_set_x - train_set_x.mean(axis=0)"""
            if gcn is not None:
                gcn = float(gcn)
                train_set_x = (train_set_x.T - train_set_x.mean(axis=1)).T
                train_set_x = (train_set_x.T / numpy.sqrt(numpy.square(train_set_x).sum(axis=1))).T
                train_set_x *= gcn
            self.x = numpy.cast['float32'](train_set_x)
            self.y = train_set_y
        else:
            test_set_x = numpy.zeros( (2*NUM_IMAGE_BATCH,2*(IMAGE_TARGET_SIZE)**2))
            dataset = data_path+'data_batch_11'
            f = open(dataset, 'rb')
            test_set= pickle.load(f)
            test_set_x[0:NUM_IMAGE_BATCH,:]=test_set['data'][:]
            test_set_y=test_set['labels']
            f.close()
            
            dataset = data_path+'data_batch_12'
            f = open(dataset, 'rb')
            test_set= pickle.load(f)
            test_set_x[NUM_IMAGE_BATCH:2*NUM_IMAGE_BATCH,:] = test_set['data']
            test_set_y += test_set['labels']
            test_set_y = numpy.asarray(test_set_y).astype('float32')
            f.close()
            
            """test_set_x = test_set_x / 255.
            other = NORB(which_set='train')
            ox = other.x
            ox /= 255.
            test_set_x = test_set_x - ox.mean(axis=0)"""
            if gcn is not None:
                gcn = float(gcn)
                test_set_x = (test_set_x.T - test_set_x.mean(axis=1)).T
                test_set_x = (test_set_x.T / numpy.sqrt(numpy.square(test_set_x).sum(axis=1))).T
                test_set_x *= gcn
            self.x = numpy.cast['float32'](test_set_x)
            self.y = test_set_y
    def get_data(self):
            """
            Returns all the data, as it is internally stored.
            The definition and format of these data are described in
            `self.get_data_specs()`.
    
            Returns
            -------
            data : numpy matrix or 2-tuple of matrices
                The data
            """
            
            if self.y is None:
                return self.x
            else:
                return (self.x, self.y)