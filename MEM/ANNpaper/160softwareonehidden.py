




#MEM160 Linear 

#onelayer with new acc caculatation mathod shrink the range of drop function within (0,2), add a shift parameter, plus 1000 loop software weights value
from multiprocessing import Pool
from tensorflow.contrib import autograph
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/home/rujie/tensorflow/MNIST_data", one_hot = True)
testimage= mnist.test.images
testlabel=mnist.test.labels

# global_step = tf.Variable(0,trainable=False)
# starter_learning_rate = 0.0009
# learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,1000,0.96,staircase=True)
learning_rate = 0.005
hl1=10
n_output=10
m=1.1
std=0.16
hm_epochs = 500
batch_size = 100

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
j = tf.placeholder(tf.float32,shape=[784,10])
k = tf.placeholder(tf.float32,shape=[784,10])
r_hl1=np.array([[0. for i in range(hl1)] for i in range(784)],dtype=np.float32)
r_output=np.array([[0. for i in range(n_output)] for i in range(hl1)],dtype=np.float32)
#w = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[784, 10]))
#b = tf.Variable(tf.constant(1,dtype=tf.float32,shape=[10]))
w_hl1 = tf.Variable(tf.random_normal([784,hl1],mean=m,stddev=std,dtype=tf.float32))
w_output = tf.Variable(tf.random_normal([hl1,n_output],mean=m,stddev=std,dtype=tf.float32))
b_hl1 = np.array([[0.01]*hl1])
b_output = np.array([[0.01]*n_output])

y_hl1 = tf.nn.relu(tf.matmul(x,w_hl1)+b_hl1)
y = tf.cast(tf.matmul(y_hl1,w_output)+b_output,dtype=tf.float32)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+1e-3), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_, logits= y))

#train = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE,momentum=0.9, use_nesterov=True)
#training = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(cross_entropy)
#training = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cross_entropy)
#training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
#training_s = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
#training = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step = global_step)
training = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range (hm_epochs):
		epoch_loss = 0
		for _ in range(int(mnist.train.num_examples/batch_size)):
			epoch_x,epoch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([training, cross_entropy], feed_dict = {x:epoch_x, y_:epoch_y})
			epoch_loss += c
		print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss,'Accuracy:',accuracy.eval({x:mnist.test.images, y_:mnist.test.labels}))
	updated_weights_hl1 = sess.run(w_hl1)
	updated_weights_output = sess.run(w_output)
	np.save('Port26_160softwareonehidden_whl1',updated_weights_hl1)
	np.save('Port26_160softwareonehidden_woutput',updated_weights_output)





