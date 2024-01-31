




from multiprocessing import Pool
from tensorflow.contrib import autograph
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/rujie/tesorflow/MEM/MNIST_data/", one_hot=True)

batch_size = 100
keep_rate = 0.95
x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32')
hiddenlayer = 1000
output = 10
batch_size = 100

hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,hiddenlayer])),
					'biases':tf.Variable(tf.random_normal([hiddenlayer]))}
output_layer = {'weights':tf.Variable(tf.random_normal([hiddenlayer,output])),
					'biases':tf.Variable(tf.random_normal([output]))}

l1 = tf.nn.relu(tf.add(tf.matmul(x,hidden_1_layer['weights']),hidden_1_layer['biases']))
l1 = tf.layers.batch_normalization(l1)
l2 = tf.add(tf.matmul(l1,output_layer['weights']),output_layer['biases'])

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= l2))
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)
correct = tf.equal(tf.argmax(l2,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
epochs = 20

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for _ in range(int(mnist.train.num_examples/batch_size)):
			epoch_x,epoch_y = mnist.train.next_batch(batch_size)
			sess.run(optimizer,feed_dict={x:epoch_x,y:epoch_y})
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		
		
		
		
		




	















