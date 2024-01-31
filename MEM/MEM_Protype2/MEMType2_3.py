



from MEM_1 import MEM
from multiprocessing import Pool
from tensorflow.contrib import autograph
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/rujie/tensorflow/MEM/MNIST_data/", one_hot=True)

batch_size = 100
keep_rate = 0.95
x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32')
hiddenlayer = 1000
output = 10
batch_size = 100

hl = {'weights':tf.Variable(tf.constant(1,dtype=tf.float32,shape=[784,hiddenlayer])),
					'biases':tf.Variable(tf.random_normal([hiddenlayer]))}
ol = {'weights':tf.Variable(tf.random_normal([hiddenlayer,output])),
					'biases':tf.Variable(tf.random_normal([output]))}


with tf.control_dependencies([MEM(hl['weights'])]):
	l1 = tf.identity(tf.nn.relu(tf.add(tf.matmul(x,tf.cast(MEM(hl['weights']),'float32')),hl['biases'])))


l1 = tf.layers.batch_normalization(l1)
l2 = tf.add(tf.matmul(l1,ol['weights']),ol['biases'])

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= l2))
optimizer = tf.train.AdamOptimizer(0.0005).minimize(cost)
correct = tf.equal(tf.argmax(l2,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
epochs = 100

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for _ in range(int(mnist.train.num_examples/batch_size)):
			epoch_x,epoch_y = mnist.train.next_batch(batch_size)
			sess.run(optimizer,feed_dict={x:epoch_x,y:epoch_y})
		#print(sess.run(hl['weights']))
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		
		
		
		
		




	















