"""
.. todo::

    WRITEME
"""
import os
import cPickle
import logging
_logger = logging.getLogger(__name__)

import numpy as np
import warnings
N = np
from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize


class CIFAR10(dense_design_matrix.DenseDesignMatrix):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : str
        One of 'train', 'test'
    center : WRITEME
    rescale : WRITEME
    gcn : float, optional
        Multiplicative constant to use for global contrast normalization.
        No global contrast normalization is applied, if None
    one_hot : WRITEME
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    toronto_prepro : WRITEME
    preprocessor : WRITEME
    """

    def __init__(self, which_set, center=False, rescale=False, gcn=None,
                 one_hot=None, start=None, stop=None, axes=('b', 0, 1, 'c'),
                 toronto_prepro = False, preprocessor = None):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)

        self.axes = axes

        # we define here:
        dtype = 'uint8'
        ntrain = 50000
        nvalid = 0  # artefact, we won't use it
        ntest = 10000

        # we also expose the following details:
        self.img_shape = (3, 32, 32)
        self.img_size = N.prod(self.img_shape)
        self.n_classes = 10
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

        # prepare loading
        fnames = ['data_batch_%i' % i for i in range(1, 6)]
        lenx = N.ceil((ntrain + nvalid) / 10000.)*10000
        x = N.zeros((lenx, self.img_size), dtype=dtype)
        y = []

        # load train data
        nloaded = 0
        for i, fname in enumerate(fnames):
            data = CIFAR10._unpickle(fname)
            x[i*10000:(i+1)*10000, :] = data['data']
            y.append(data['labels'])
            nloaded += 10000
            if nloaded >= ntrain + nvalid + ntest:
                break

        # load test data
        data = CIFAR10._unpickle('test_batch')
        y.append(data['labels'])
        y = N.concatenate(y)
        # process this data
        Xs = {'train': x[0:ntrain],
              'test': data['data'][0:ntest]}

        Ys = {'train': y[0:ntrain],
              'test': y[ntrain:ntrain+ntest]}

        X = N.cast['float32'](Xs[which_set])
        y = Ys[which_set]

        if isinstance(y, list):
            y = np.asarray(y).astype(dtype)

        max_labels = 10
        if one_hot is not None:
            warnings.warn("the `one_hot` parameter is deprecated. To get "
                          "one-hot encoded targets, request that they "
                          "live in `VectorSpace` through the `data_specs` "
                          "parameter of MNIST's iterator method. "
                          "`one_hot` will be removed on or after "
                          "September 20, 2014.", stacklevel=2)

        if center:
            X -= 127.5
        self.center = center

        if rescale:
            X /= 127.5
        self.rescale = rescale

        if toronto_prepro:
            assert not center
            assert not gcn
            X = X / 255.
            if which_set == 'test':
                other = CIFAR10(which_set='train')
                oX = other.X
                oX /= 255.
                X = X - oX.mean(axis=0)
            else:
                X = X - X.mean(axis=0)
        self.toronto_prepro = toronto_prepro

        self.gcn = gcn
        if gcn is not None:
            gcn = float(gcn)
            X = global_contrast_normalize(X, scale=gcn)

        if start is not None:
            # This needs to come after the prepro so that it doesn't
            # change the pixel means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop, :]
            assert X.shape[0] == y.shape[0]

        #view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3),
                                                                  #axes)

        super(CIFAR10, self).__init__(X=X, y=y,y_labels=self.n_classes)

        assert not np.any(np.isnan(self.X))

        if preprocessor:
            preprocessor.apply(self)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        # assumes no preprocessing. need to make preprocessors mark the
        # new ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i, :] /= np.abs(rval[i, :]).max()
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval, -1., 1.)

        return rval

    def adjust_to_be_viewed_with(self, X, orig, per_example=False):
        """
        .. todo::

            WRITEME
        """
        # if the scale is set based on the data, display X oring the
        # scale determined by orig
        # assumes no preprocessing. need to make preprocessors mark
        # the new ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'rescale'):
            self.rescale = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i, :] /= np.abs(orig[i, :]).max()
            else:
                rval /= np.abs(orig).max()
            rval = np.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        if not self.rescale:
            rval /= 127.5

        rval = np.clip(rval, -1., 1.)

        return rval

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return CIFAR10(which_set='test', center=self.center,
                       rescale=self.rescale, gcn=self.gcn,
                       toronto_prepro=self.toronto_prepro,
                       axes=self.axes)

    @classmethod
    def _unpickle(cls, file):
        """
        .. todo::

            What is this? why not just use serial.load like the CIFAR-100
            class? Whoever wrote it shows up as "unknown" in git blame.
        """
        from pylearn2.utils import string_utils
        fname = os.path.join(string_utils.preprocess('/home/shizenglin/masterwork/data/'),
                             'cifar10', 'cifar-10-batches-py', file)
        if not os.path.exists(fname):
            raise IOError(fname+" was not found. You probably need to "
                          "download the CIFAR-10 dataset by using the "
                          "download script in "
                          "pylearn2/scripts/datasets/download_cifar10.sh "
                          "or manually from "
                          "http://www.cs.utoronto.ca/~kriz/cifar.html")
        fname = cache.datasetCache.cache_file(fname)

        _logger.info('loading file %s' % fname)
        fo = open(fname, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict
