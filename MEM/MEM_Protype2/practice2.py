




from multiprocessing import Pool
from tensorflow.contrib import autograph
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tensorflow.examples.tutorials.mnist import input_data

config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#sess = tf.Session(config=config)

mnist = input_data.read_data_sets("/home/rujie/tensorflow/MEM/MNIST_data/", one_hot=True)

batch_size = 100
keep_rate = 0.95
x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32')


dense1 = tf.layers.dense(inputs=x, units=1000, activation="relu", use_bias=True,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
						bias_initializer=tf.zeros_initializer())
#dense1 = tf.nn.dropout(dense1, keep_rate)					
dense2 = tf.layers.dense(inputs=dense1, units=10, activation=None, use_bias=True,
						kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
						bias_initializer=tf.zeros_initializer())

cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= dense2))
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)
correct = tf.equal(tf.argmax(dense2,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
epochs = 20

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		for _ in range(int(mnist.train.num_examples/batch_size)):
			epoch_x,epoch_y = mnist.train.next_batch(batch_size)
			sess.run(optimizer,feed_dict={x:epoch_x,y:epoch_y})
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))







