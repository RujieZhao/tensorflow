
# One layer training module
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
testimage= mnist.test.images
print('testimage_shape:',testimage.shape)
testlabel=mnist.test.labels
print('testlabel_shape:',testlabel.shape)
hl1=10
n_output=10

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#global_step = tf.Variable(0,trainable=False)
#starter_learning_rate=0.05
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,500,0.96,staircase=True)

w_hl1 = tf.Variable(tf.random_normal([784,hl1],mean=1.1,stddev=0.16,dtype=tf.float32))
w_output = tf.Variable(tf.random_normal([hl1,n_output],mean=1.1,stddev=0.16,dtype=tf.float32))
b_hl1 = np.array([[0.01]*hl1])
b_output = np.array([[0.01]*n_output])

LEARNING_RATE = 0.05
TRAIN_STEPS = 2000

y_hl1 = tf.nn.relu(tf.matmul(x,w_hl1)+b_hl1)
y = tf.cast(tf.matmul(y_hl1,w_output)+b_output,dtype=tf.float32)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y_, logits= y))
#training = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
training = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
#training = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(cross_entropy)
#training = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

xacc=range(0,TRAIN_STEPS+1,1)
acc= np.array(np.array([0.]*(TRAIN_STEPS+1),dtype=np.float32))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())		
	for i in range(1,TRAIN_STEPS+1):		
		sess.run(training, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
		sess.run(tf.assign(w_output,tf.clip_by_value(w_output,0.,2.)))
		sess.run(tf.assign(w_hl1,tf.clip_by_value(w_hl1,0.,2.)))
		if i % 100 == 0:
			print('i:',i)
			print(sess.run(accuracy,feed_dict={x:testimage,y_:testlabel}))
'''
		acc[i] = sess.run(accuracy,feed_dict={x:testimage,y_:testlabel})*100
		print(acc[i])
		if i%500 == 0:
			plt.plot(xacc,acc,label = 'GPU', linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=5)
			plt.yticks(np.linspace(0,100,21))
			plt.xticks(np.linspace(0,TRAIN_STEPS,TRAIN_STEPS/100+1))
			plt.xlabel('Training times')
			plt.ylabel('Accuracy%')
			plt.title('Trendency of Accuracy GPU')
			plt.grid()
			plt.legend()
			#plt.savefig()
			plt.show()
			plt.close()
'''			
			
			
			
			
			
			
			








