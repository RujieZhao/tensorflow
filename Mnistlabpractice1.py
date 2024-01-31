'''
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32',[None,10])
w = tf.Variable(tf.random.normal([784,10],dtype=tf.float32))
b = tf.Variable(tf.constant(1,dtype=tf.float32,shape=[10]))
output = tf.nn.relu(tf.matmul(x,w)+b)

with tf.Session() as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	for s in range(10):
		#weight = r[:,s]-w_o[:,s]
		weight = sess.run(w)[:,s]
		plt.suptitle('weights random nomal initialization',fontsize=15)
		plt.subplot(2,5,s+1)
		plt.title(s)
		plt.imshow(weight.reshape([28,28]),cmap=plt.get_cmap('seismic'))#gray_r
		frame1=plt.gca()
		frame1.axes.get_xaxis().set_visible(False)
		frame1.axes.get_yaxis().set_visible(False)
	plt.show()
'''



#print('the length of %s is %d' % ('hello',5))	
	
	
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32',[None,10])
w = tf.Variable(tf.random.normal([784,10],dtype=tf.float32))
b = tf.Variable(tf.constant(1,dtype=tf.float32,shape=[10]))
output = tf.nn.relu(tf.matmul(x,w)+b)

global_step = tf.Variable(0,trainable=False)
starter_learning_rate=0.05
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,300,0.96,staircase=True)

		
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= output))		
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step=global_step)

correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
hm_epochs = 2000

with tf.Session() as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	epoch_loss=0
	for i in range(1,hm_epochs+1):		
		print('global_step_before:',global_step.eval())
		_,c=sess.run([optimizer,cost],feed_dict={x:mnist.train.images, y:mnist.train.labels})
		epoch_loss += c
		
		print('Accuracy:',accuracy.eval({x:mnist.train.images,y:mnist.train.labels}))			
		print('global_step:',global_step.eval())
		print('LR:',learning_rate.eval())
		#print('i:',i)
	
	
	
	
	
	
	
	
	
