


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/rujie/tensorflow/MNIST_data/", one_hot=True)

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32',[None,10])
w = tf.Variable(tf.random.normal([784,10],dtype=tf.float32))
b = tf.Variable(tf.constant(1,dtype=tf.float32,shape=[10]))
output = tf.nn.relu(tf.matmul(x,w)+b)
		
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= output))		
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
hm_epochs = 1000

with tf.Session() as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	epoch_loss=0
	for i in range(1,hm_epochs+1):		
		_,c=sess.run([optimizer,cost],feed_dict={x:mnist.train.images, y:mnist.train.labels})
		epoch_loss += c
		if i %10 == 0:
			print('Epoch:',i,'completed out of', hm_epochs, 'loss:', epoch_loss)
			print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
		if i==hm_epochs:
			weight=sess.run(w)
			np.save('w_new.npy',weight)





