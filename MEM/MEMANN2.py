
#hiden layer 1

import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/rujie/tensorflow/MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#j = tf.placeholder(tf.float32,shape=[784,10])
#k = tf.placeholder(tf.float32,shape=[784,10])

hl1=1000
n_output=10

w_hl1 = tf.Variable(tf.random_normal([784,hl1],mean,stddev))
w_output = tf.Variable(tf.random_normal([hl1,n_output],mean,stddev))
b_hl1 = tf.Variable(tf.random_normal([hl1],mean,stddev))
b_output = tf.Variable(tf.random_normal([n_output],mean,stddev))

y_hl1=tf.nn.relu(tf.matmul(x,w_hl1)+b_hl1)
y = tf.matmul(y_hl1,w)+b_output
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y_, logits= y))

mean =0
stddev=1
LEARNING_RATE = 0.5
TRAIN_STEPS = 2
limit = 2
m =0.2
Points=1
#initializer = tf.contrib.layers.xavier_initilizer()
Ry=2.3551 #11.76148
Ra= -2.421 #-12.09054
Rt=23.0499 #23.05015

Dy = 0.6959 #0.77
Da = 3.0708 #232.57604
Dt = 12.2756 #8.69255

def MEM_ini(w):
	N = tf.round(-Rt*(tf.math.log(tf.div((tf.clip_by_value(w,m,(Ry-1e-3))-Ry),Ra)))*Points)
	return Ra*tf.exp(-tf.div(N,(Points*Rt)))+Ry

def MEM_R(w,k,j):
	with tf.device('/gpu:0'):
		n_1 = tf.round(-Rt*(tf.math.log(tf.div((tf.clip_by_value(k,m,(Ry-1e-3))-Ry),Ra)))*Points)
		n_0 = tf.round(-Rt*(tf.math.log(tf.div((tf.clip_by_value(j,m,(Ry-1e-3))-Ry),Ra)))*Points)
		n_diff_1 = n_1-n_0
		return tf.assign(w,tf.cond(n_diff_1>limit,lambda:Ra*tf.exp(-tf.div((n_0+limit),(Points*Rt)))+Ry,lambda:Ra*tf.exp(-tf.div(n_1,(Points*Rt)))+Ry))

def MEM_D(w,k,j):	
	with tf.device('/gpu:1'):	
		n_1 = tf.round(-Dt*(tf.math.log((tf.clip_by_value(k,(Dy+1e-3),3.31)-Dy)/Da))*Points)
		n_0 = tf.round(-Dt*(tf.math.log((tf.clip_by_value(j,(Dy+1e-3),3.31)-Dy)/Da))*Points)
		n_diff_2=n_1-n_0
		return tf.assign(w,tf.cond(n_diff_2>limit,lambda:Da*tf.exp(-tf.div((n_0+limit),(Points*Dt)))+Dy,lambda:Da*tf.exp(-tf.div(n_1,(Points*Dt)))+Dy))	    


testimage= mnist.test.images
testlabel= mnist.test.labels

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())	
	#q = sess.run(w)
	sess.run(tf.assign(w_hl1, MEM_ini(w_hl1)))
	j_w_hl1 = sess.run(w_hl1)
	sess.run(tf.assign(w_output, MEM_ini(w_output)))
	j_w_output= sess.run(w_output)
	sess.run(tf.assign(b_hl1,MEM_ini(b_hl1)))
	j_b_hl1 = sess.run(b_hl1)
	sess.run(tf.assign(b_output,MEM_ini(b_output)))
	j_b_output = sess.run(b_output)
	
	training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	for i in range(0,TRAIN_STEPS):
		#print('w_old:',q)
		sess.run(training, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
		k_w_hl1 = sess.run(w_hl1)
		k_w_output = sess.run(w_output)
		k_b_hl1 = sess.run(b_hl1)
		k_b_ouput = sess.run(b_ouput)
		
		for a in range(0,784):
			for b in range(0,1000):
				w_diff = k_w_hl1[b,a]-j_w_hl1[b,a]
				result1=tf.case({tf.greater(w_diff,0):lambda:MEM_R(w[b,a],k_w_hl1[b,a],j_w_hl1[b,a]),tf.less(w_diff,0):lambda:MEM_D(w[b,a],k_w_hl1[b,a],j_w_hl1[b,a])},default=lambda:0.,exclusive=True)
				print('w['+str(a)+','+str(b)+']:')	
		
		with tf.device('/gpu:2'):			
			w_new = sess.run(w)
			y_new = np.dot(testimage,w_new)+b
			prediction = np.equal(np.argmax(y_new,1),np.argmax(testlabel,1))
			np_accuracy = np.float32(np.mean(np.cast[float](prediction)))
			print(np_accuracy)
			print('Training Step:'+ str(i)+'Accuracy= '+str(sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})) +'Loss= '+str(sess.run(cross_entropy, {x: mnist.train.images, y_: mnist.train.labels})))
		j=sess.run(w)
			
			
	
			
			
			
			
			
