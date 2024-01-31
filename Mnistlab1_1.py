
# one layer ANN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32',[None,10])
#w = tf.Variable(tf.random.normal([784,10],mean=1.1,stddev=0.16,dtype=tf.float32))
w = tf.Variable(tf.random_uniform([784,10],-3,3,dtype = tf.int32))
b = tf.Variable(tf.constant(2,dtype=tf.float32,shape=[10]))
output = tf.nn.relu(tf.matmul(x,tf.cast(w,tf.float32)+b)
		
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= output))		
optimizer = tf.train.AdamOptimizer(0.05).minimize(cost)

correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
#hm_epochs = 2000

#xacc=range(0,hm_epochs+1,1)	
#ACC= np.array([0.]*(hm_epochs+1),dtype=np.float32)
with tf.Session(config=config) as sess:
	#sess.run(tf.initialize_all_variables())
	#print('xacc:',xacc)
	sess.run(tf.global_variables_initializer())
	#epoch_loss=0
	print(w.eval())
	for i in range(1,hm_epochs+1):		
		#print('i:',i)
		_,c=sess.run([optimizer,cost],feed_dict={x:mnist.train.images, y:mnist.train.labels})
		epoch_loss = c
		print('loss:',epoch_loss)
		print(accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
		if i == 5:
			print(w.eval())
			break
		#if i %1 == 0:
			#print('Epoch:',i,'completed out of', hm_epochs, 'loss:', epoch_loss)
			#print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
		#ACC[i]=accuracy.eval({x:mnist.test.images,y:mnist.test.labels})*100

'''
		if i % 10 == 0:							
			plt.plot(xacc,ACC,label='Points2000',linewidth=1,color='r',marker='o',markerfacecolor='blue',markersize=5)
			#plt.ylim(0,100)
			#plt.xlim(0,i+1)			
			plt.yticks(np.linspace(0,100,21))
			plt.xticks(np.linspace(0,hm_epochs,hm_epochs/5+1))
			plt.xlabel('Training times')
			plt.ylabel('Accuracy%')
			plt.title('Trendency of Accuracy Port28')
			plt.grid()
			plt.legend()
			#plt.savefig('Port24_loop('+str(i)+').png',dpi=400)
			plt.show()	
			plt.close()	
'''

'''	
	for epoch in range(hm_epochs):
		epoch_loss = 0		
		for _ in range(int(mnist.train.num_examples/batch_size)):
			epoch_x, epoch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
			epoch_loss += c
 #the first print is one indence tab with the first for loop, so it will just occur 10 times
		print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

	correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1)) 
	accuracy = tf.reduce_mean(tf.cast(correct,'float'))

	print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

#train_neural_network(x)
'''











