
# the second layer doesnt work

import math
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
testimage= mnist.test.images[0:100,:]
print('testimage_shape:',testimage.shape)
testlabel=mnist.test.labels[0:100,:]
print('testlabel_shape:',testlabel.shape)
hl1=10
n_output=10

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

w_hl1 = tf.Variable(tf.random_normal([784,hl1],mean=0.92,stddev=0.22,dtype=tf.float32))
w_output = tf.Variable( tf.constant(0.1,dtype=tf.float32,shape=[hl1,n_output]),trainable=True)
#w_output = tf.constant(1,dtype=tf.float32,shape=[hl1,n_output])
#w_output = tf.Variable(tf.random_normal([hl1,n_output],mean=0.95,stddev=0.2,dtype=tf.float32),trainable=False)

#LEARNING_RATE =0.0357
LEARNING_RATE =0.05
hm_epoch=99
TRAIN_STEPS = 1000

y_hl1 = tf.nn.relu(tf.matmul(x,w_hl1))/784.
y = tf.cast(tf.matmul(y_hl1,w_output)/10,dtype=tf.float32)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_, logits= y))
training = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())		
	for i in range(0,TRAIN_STEPS):		
		#print('j_output_i('+str(i)+')',sess.run(w_output))
		sess.run(training, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
		#print('k_output_i('+str(i)+'):',sess.run(w_output))
		#print('k_hl1_i('+str(i)+')[230:250,:]:',sess.run(w_hl1)[230:250,:])
		if i%1 == 0:
			#print(sess.run(w_output))
			print('Accuracy_i('+str(i)+'):',sess.run(accuracy,feed_dict={x:testimage,y_:testlabel}))	
			r_hl1 = sess.run(w_hl1)
			#print('r_hl1[0:2,:]:',r_hl1[0:2,:])
			r_1 = sess.run(w_output)
			#print('r_1[0:2,:]:',r_1[0:2,:])
			y_hl1 = np.dot(testimage,r_hl1)
			y_output = np.dot(y_hl1,r_1)
			prediction = np.equal(np.argmax(y_output,1),np.argmax(testlabel,1))
			np_accuracy = np.float32(np.mean(np.cast[float](prediction)))
			print('np_accuracy_i('+str(i+1)+'):',np_accuracy)
















