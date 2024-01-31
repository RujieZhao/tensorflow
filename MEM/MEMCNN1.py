import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


n_classes = 10
batch_size = 128
x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32',[None,10])

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

w_conv1=tf.Variable(tf.random.normal([5,5,1,32],dtype=tf.float32)) 				  
w_conv2=tf.Variable(tf.random.normal([5,5,32,64],dtype=tf.float32))
w_fc=tf.Variable(tf.random.normal([7*7*64,1024],dtype=tf.float32))
w_out=tf.Variable(tf.random.normal([1024,n_classes],dtype=tf.float32))
#b_conv1=tf.Variable(tf.random.normal([32],dtype=tf.float32)) 				  
#b_conv2=tf.Variable(tf.random.normal([64],dtype=tf.float32))
b_fc=tf.Variable(tf.random.normal([1024],dtype=tf.float32))
b_out=tf.Variable(tf.random.normal([n_classes],dtype=tf.float32))

x_new =tf.reshape(x,shape=[-1,28,28,1])
conv1 =conv2d(x_new,w_conv1)
conv1 = maxpool2d(conv1)

conv2 =conv2d(conv1,w_conv2)
conv2 = maxpool2d(conv2)

fc = tf.reshape(conv2,[-1,7*7*64])
fc = tf.nn.relu(tf.matmul(fc,w_fc)+b_fc)

fc = tf.nn.dropout(fc, keep_rate)

output = tf.matmul(fc,w_out)+b_out
		
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= output))		
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
hm_epochs = 10

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	epoch_loss=0
	for epoch in range(hm_epochs):
		epoch_loss = 0		
		for _ in range(int(mnist.train.num_examples/batch_size)):
			epoch_x, epoch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
			epoch_loss += c
		print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
		print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

	


'''
	for i in range(10):
		weight = sess.run(w_out)[:,i]-j[:,i]
		plt.subplot(2,5,i+1)
		plt.suptitle(Accuracy)
		plt.title(i)
		plt.imshow(weight.reshape([32,32]),cmap=plt.get_cmap('seismic'))
		frame1=plt.gca()
		frame1.axes.get_xaxis().set_visible(False)
		frame1.axes.get_yaxis().set_visible(False)
	plt.show()

	for i in range(1,hm_epochs+1):		
		_,c=sess.run([optimizer,cost],feed_dict={x:mnist.train.images, y:mnist.train.labels})
		epoch_loss += c
		if i %5 == 0:
			print('Epoch:',i,'completed out of', hm_epochs, 'loss:', epoch_loss)
			print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
'''







