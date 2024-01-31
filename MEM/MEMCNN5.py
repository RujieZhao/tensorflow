


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/rujie/tensorflow/MNIST_data", one_hot=True)

n_classes = 10
batch_size = 128
con=0.5
m=1.2
st=0.2
Points=300
#initializer = tf.contrib.layers.xavier_initilizer()
Ry=2.3551 #11.76148
Ra= -2.421 #-12.09054
Rt=23.0499 #23.05015

Dy = 0.6959 #0.77
Da = 3.0708 #232.57604
Dt = 12.2756 #8.69255

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32',[None,10])

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
def w_initializer(w):
	w_1 = tf.round(-Rt*(tf.math.log(tf.div((w-Ry),Ra)))*Points)
	w_2 = Ra*tf.exp(-tf.div(w_1,(Points*Rt)))+Ry
	return w_2

w_conv1=tf.Variable(tf.random.normal(shape=[5,5,1,32],mean=m,stddev=st,dtype=tf.float32)*tf.math.sqrt(1/32))	  
w_conv2=tf.Variable(tf.random.normal(shape=[5,5,32,64],mean=m,stddev=st,dtype=tf.float32)*tf.math.sqrt(1/64))
w_fc=tf.Variable(tf.random.normal(shape=[7*7*64,1024],mean=m,stddev=st,dtype=tf.float32)*tf.math.sqrt(1/1024))
w_out=tf.Variable(tf.random.normal(shape=[1024,n_classes],mean=m,stddev=st,dtype=tf.float32)*tf.math.sqrt(1/n_classes))
b_fc=tf.Variable(tf.random.normal(shape=[1024],mean=m,stddev=st,dtype=tf.float32)*tf.math.sqrt(1/1024))
b_out=tf.Variable(tf.random.normal(shape=[n_classes],mean=m,stddev=st,dtype=tf.float32)*tf.math.sqrt(1/n_classes))

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
#optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
#optimizer = tf.train.AdadeltaOptimizer(1.2).minimize(cost)
correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
hm_epochs = 30
limit = 20.
m =0.2

def MEM_R(k,j):
	with tf.device('/gpu:0'):
		n_1 = tf.round(-Rt*(tf.math.log(tf.div((tf.clip_by_value(k,m,(Ry-1e-3))-Ry),Ra)))*Points)
		n_0 = tf.round(-Rt*(tf.math.log(tf.div((tf.clip_by_value(j,m,(Ry-1e-3))-Ry),Ra)))*Points)
		n_diff_1 = n_1-n_0
		return tf.assign(k,tf.cond(n_diff_1>limit,lambda:Ra*tf.exp(-tf.div((n_0+limit),(Points*Rt)))+Ry,lambda:Ra*tf.exp(-tf.div(n_1,(Points*Rt)))+Ry))


def MEM_D(k,j):	
	with tf.device('/gpu:1'):	
		n_1 = tf.round(-Dt*(tf.math.log((tf.clip_by_value(k,(Dy+1e-3),3.31)-Dy)/Da))*Points)
		n_0 = tf.round(-Dt*(tf.math.log((tf.clip_by_value(j,(Dy+1e-3),3.31)-Dy)/Da))*Points)
		n_diff_2=n_1-n_0
		return tf.assign(k,tf.cond(n_diff_2>limit,lambda:Da*tf.exp(-tf.div((n_0+limit),(Points*Dt)))+Dy,lambda:Da*tf.exp(-tf.div(n_1,(Points*Dt)))+Dy))


with tf.Session() as sess:
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())	
	epoch_loss=0
	
	sess.run(tf.assign(w_conv1,w_initializer(w_conv1)))
	sess.run(tf.assign(w_conv2,w_initializer(w_conv2)))
	sess.run(tf.assign(w_fc,w_initializer(w_fc)))
	sess.run(tf.assign(w_out,w_initializer(w_out)))
	sess.run(tf.assign(b_fc,w_initializer(b_fc)))
	sess.run(tf.assign(b_out,w_initializer(b_out)))

	w_conv1_j=sess.run(w_conv1)
	w_conv2_j=sess.run(w_conv2)
	w_fc_j=sess.run(w_fc)
	w_out_j=sess.run(w_out)
	b_fc_j=sess.run(b_fc)
	b_out_j=sess.run(b_out)
	
	for epoch in range(hm_epochs):
		epoch_loss = 0		
		for _ in range(int(mnist.train.num_examples/batch_size)):
			epoch_x, epoch_y = mnist.train.next_batch(batch_size)
			_, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
			epoch_loss += c
			
		#with tf.device('/gpu:0'):
		for a in range(0,5):
			for b in range(0,5):
				for d in range(0,1):
					#with tf.device('/gpu:1'):
					w_conv1_diff = w_conv1[a,b,0,d]-w_conv1_j[a,b,0,d]
					print('w_conv1_diff['+str(a)+','+str(b)+',0,'+str(d)+']',sess.run(w_conv1_diff))
					result1=tf.case({tf.greater(w_conv1_diff,0):lambda:MEM_R(w_conv1[a,b,0,d],w_conv1_j[a,b,0,d]),tf.less(w_conv1_diff,0):lambda:MEM_D(w_conv1[a,b,0,d],w_conv1_j[a,b,0,d])},default=None,exclusive=True)
					sess.run(result1)
					
		for a in range(0,1):
			for b in range(0,1):
				for c in range(0,1):
					for d in range(0,1):
						w_conv2_diff = w_conv2[a,b,c,d]-w_conv2_j[a,b,c,d]
						print('w_conv2_diff['+str(a)+','+str(b)+','+str(c)+str(d)+']',sess.run(w_conv2_diff))
						result2=tf.case({tf.greater(w_conv2_diff,0):lambda:MEM_R(w_conv2[a,b,c,d],w_conv2_j[a,b,c,d]),tf.less(w_conv2_diff,0):lambda:MEM_D(w_conv2[a,b,c,d],w_conv2_j[a,b,c,d])},default=None,exclusive=True)
						sess.run(result2)
		
		for a in range(0,1):
			for b in range(0,1):
				w_fc_diff = w_fc[a,b]-w_fc_j[a,b]
				print('w_fc_diff['+str(a)+','+str(b)+']',sess.run(w_fc_diff))
				result3=tf.case({tf.greater(w_fc_diff,0):lambda:MEM_R(w_fc[a,b],w_fc_j[a,b]),tf.less(w_fc_diff,0):lambda:MEM_D(w_fc[a,b],w_fc_j[a,b])},default=None,exclusive=True)
				sess.run(result3)
				
		for a in range(0,1):
			for b in range(0,1):
				w_out_diff = w_out[a,b]-w_out_j[a,b]
				print('w_out_diff['+str(a)+','+str(b)+']',sess.run(w_out_diff))
				result4=tf.case({tf.greater(w_out_diff,0):lambda:MEM_R(w_out[a,b],w_out_j[a,b]),tf.less(w_out_diff,0):lambda:MEM_D(w_out[a,b],w_out_j[a,b])},default=None,exclusive=True)
				sess.run(result4)
				
		for a in range(0,1):
			b_fc_diff = b_fc[a]-b_fc_j[a]
			print('b_fc_diff['+str(a)+','+str(b)+']',sess.run(b_fc_diff))
			result5=tf.case({tf.greater(b_fc_diff,0):lambda:MEM_R(b_fc[a],b_fc_j[a]),tf.less(b_fc_diff,0):lambda:MEM_D(b_fc[a],b_fc_j[a])},default=None,exclusive=True)
			sess.run(result5)
				
		for a in range(0,1):
			b_out_diff = b_out[a]-b_out_j[a]
			print('b_out_diff['+str(a)+','+str(b)+']',sess.run(b_out_diff))
			result6=tf.case({tf.greater(b_out_diff,0):lambda:MEM_R(b_fc[a],b_fc_j[a]),tf.less(b_out_diff,0):lambda:MEM_D(b_out[a],b_out_j[a])},default=None,exclusive=True)
			sess.run(result6)		
				
		print('Epoch', epoch, 'completed out of', hm_epochs,'loss:', epoch_loss,'Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
		
		
		
		
		
		
