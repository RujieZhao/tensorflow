




import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/rujie/tensorflow/MNIST_data/", one_hot=True)

x_train = mnist.train.images
y_train = mnist.train.labels

def display_digit(num):
    #print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
#                     show the gray pic 'gray_r' 'seismic' 'OrRd_r'   
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('MNIST Example: %d' % (num),dpi=900)
    plt.show()

display_digit(0)




