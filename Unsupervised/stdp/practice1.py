import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import matplotlib.cm as cmap
import time
import os.path
import scipy
import pickle
#import brain_no_units
import brain as b
from struct import unpack, pack,calcsize
from brain import *
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# a = pack('2I3sI',1220000,34,b'abc',56)
# b = unpack('2I3sI',a)

# #print(a)
# print(b)
# print(calcsize('2I3sI'))

train_images = mnist.train.images.shape
train_labels = mnist.train.labels.shape
test_images = mnist.test.images.shape
test_labels = mnist.test.labels.shape
print(train_images ,'\n',train_labels,'\n', test_images,'\n', test_labels)





# MNIST_data_path = './MNIST_data/'
# images = open(MNIST_data_path+'train-images.idx3-ubyte','rb')
# labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
# a = unpack(images.read(4))

# #a = unpack('>I', images.read(8))[0]
# #b = unpack('>I',labels.read(4))[0]
# print(a)



