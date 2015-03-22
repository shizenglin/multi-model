# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 11:55:04 2014

@author: shizenglin
"""
"""import theano
import numpy
import theano.tensor as T
def stochastic_select(rng,a,b):
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    mask = srng.random_integers(size=a.shape)    
    output = T.switch(mask,a,b)
    return(output)
    
rng=numpy.random.RandomState(89677)
a=T.dmatrix('a')
b=T.dmatrix('b')
c=stochastic_select(rng,a,b)

f = theano.function([a,b],c)
print f([[1,1,1,1],[1,1,1,1]],[[2,2,2,2],[2,2,2,2]])
print f([[1,1,1,1],[1,1,1,1]],[[2,2,2,2],[2,2,2,2]])
print f([[1,1,1,1],[1,1,1,1]],[[2,2,2,2],[2,2,2,2]])
print f([[1,1,1,1],[1,1,1,1]],[[2,2,2,2],[2,2,2,2]])"""
from svhn import SVHN

trainsets = SVHN(which_set='splitted_train')#
testsets = SVHN(which_set='test')
train_set_x, train_set_y = trainsets.get_data()
test_set_x, test_set_y = testsets.get_data()
print train_set_x.shape
print train_set_y.shape
print test_set_x.shape
print test_set_y.shape
print test_set_y[1:10]
"""from norb import NORB

trainsets = NORB(which_set='train')
testsets = NORB(which_set='test')
train_set_x, train_set_y = trainsets.get_data()
test_set_x, test_set_y = testsets.get_data()
print train_set_x[58319,:]
print train_set_y.shape
print test_set_x.shape
print test_set_y.shape
from cifar10 import CIFAR10

trainsets = CIFAR10(which_set='train')
testsets = CIFAR10(which_set='test')
train_set_x, train_set_y = trainsets.get_data()
test_set_x, test_set_y = testsets.get_data()
print train_set_x[49999,:]
print train_set_y.shape
print test_set_x.shape
print test_set_y.shape
from mnist import MNIST
trainsets = MNIST(which_set='train')
testsets = MNIST(which_set='test')
train_set_x, train_set_y = trainsets.get_data()
test_set_x, test_set_y = testsets.get_data()
print train_set_x[59999,:]
print train_set_y.shape
print test_set_x.shape
print test_set_y.shape"""