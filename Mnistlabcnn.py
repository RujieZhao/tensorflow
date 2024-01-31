import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


n_classes = 10
batch_size = 128
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def convolutional_neural_network(x):
	weights = {'w_conv1':tf.Variable(tf.random_normal([5,5,1,32])), 				  
		   'w_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
		   'w_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
		   'out':tf.Variable(tf.random_normal([1024,n_classes]))}
	biases = {'b_conv1':tf.Variable(tf.random_normal([32])), 				  
		  'b_conv2':tf.Variable(tf.random_normal([64])),
		  'b_fc':tf.Variable(tf.random_normal([1024])),
		  'out':tf.Variable(tf.random_normal([n_classes]))}

	x =tf.reshape(x,shape=[-1,28,28,1])
	conv1 =conv2d(x,weights['w_conv1'])
	conv1 = maxpool2d(conv1)

	conv2 =conv2d(conv1,weights['w_conv2'])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2,[-1,7*7*64])
	fc = tf.nn.relu(tf.matmul(fc,weights['w_fc'])+biases['b_fc'])

	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc,weights['out'])+biases['out']

	return output

#def train_neural_network(x):

prediction = convolutional_neural_network(x)		
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= prediction))		
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)
hm_epochs = 100000


with tf.Session() as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	
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





