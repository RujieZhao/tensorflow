#Tensorboard


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/rujie/tensorflow/MNIST_data/", one_hot=True)

n_classes = 10
batch_size = 128

with tf.name_scope("input"):
	x = tf.placeholder('float32', [None, 784],name="images")
	y = tf.placeholder('float32',[None,10],name="labels")

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,w,name="conv"):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x,name="maxpool"):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

with tf.name_scope("parameter"):

	w_conv1=tf.Variable(tf.random.normal([5,5,1,32],dtype=tf.float32),name="w_conv1") 				  
	tf.summary.histogram("weight1",w_conv1)
	w_conv2=tf.Variable(tf.random.normal([5,5,32,64],dtype=tf.float32),name="w_conv2")
	tf.summary.histogram("weight2",w_conv2)
	w_fc=tf.Variable(tf.random.normal([7*7*64,1024],dtype=tf.float32),name="w_fc")
	tf.summary.histogram("weight3",w_fc)
	w_out=tf.Variable(tf.random.normal([1024,n_classes],dtype=tf.float32),name="w_out")
	tf.summary.histogram("weight4",w_out)
	b_fc=tf.Variable(tf.random.normal([1024],dtype=tf.float32),name="b_fc")
	tf.summary.histogram("bias1",b_fc)
	b_out=tf.Variable(tf.random.normal([n_classes],dtype=tf.float32),name="b_out")
	tf.summary.histogram("bias2",b_out)
	
with tf.name_scope('input_shape'):	
	x_new =tf.reshape(x,shape=[-1,28,28,1])
	tf.summary.image('input_matrix',x_new,10)

conv1 =conv2d(x_new,w_conv1,"conv1")
conv1 = maxpool2d(conv1,"maxpool1")

conv2 =conv2d(conv1,w_conv2,"conv2")
conv2 = maxpool2d(conv2,"maxpool2")

fc = tf.reshape(conv2,[-1,7*7*64])
fc = tf.nn.relu(tf.matmul(fc,w_fc)+b_fc,name="fc")
tf.summary.histogram("hide_layer_1",fc)
fc = tf.nn.dropout(fc, keep_rate)
output = tf.math.add(tf.matmul(fc,w_out),b_out,name="output")
tf.summary.histogram("activation",output)

with tf.name_scope("cost"):		
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= output))		
	tf.summary.scalar('cost',cost)
	
with tf.name_scope("optimizer"):
	optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

with tf.name_scope("accuracy"):
	correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1)) 
	accuracy = tf.reduce_mean(tf.cast(correct,'float'))
	tf.summary.scalar('accuracy',accuracy)


hm_epochs = 6
with tf.Session() as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	merged_summary=tf.summary.merge_all()
	writer = tf.summary.FileWriter("example2")		
		
	epoch_loss=0
	for learning_rate in [1e-3,1e-4,1e-5]:
		
		for epoch in range(hm_epochs):
			epoch_loss = 0				
			for i in range(1,int(mnist.train.num_examples/batch_size)+1):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				if epoch % 2 ==0:
					s=sess.run(merged_summary,feed_dict={x:epoch_x, y:epoch_y})
					writer.add_summary(s,i)
				_, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
				epoch_loss += c 
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
			print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
	
	writer.add_graph(sess.graph)







