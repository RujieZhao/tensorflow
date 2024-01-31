



import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
np.set_printoptions(threshold=np.inf)
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth=True
mnist = input_data.read_data_sets("/home/rujie/tensorflow/MNIST_data/",one_hot=True)

def np_spiky(x):
	x = np.int(x)
	#print('x from np_spiky:',x)
	#y = math.pow(x,2)*1.5
	y = x
	#print('y from np_spiky:',y)
	return y
v = np.vectorize(np_spiky)
vd = lambda x:np.float32(v(x))
print(np_spiky(2.5),type(np_spiky(2.5)),type(v(2.5)),vd(2.5))

def tf_d_spiky(op,grad):
	x = op.inputs[0]
	n_gr = tf.to_float(tf.multiply(x,3.))
	return n_gr*grad

def py_func(func,inp,tout,stateful=True,name=None,grad=None):
	rnd_name="PyFuncGrad"+str(np.random.randint(0,1e+10))
	tf.RegisterGradient(rnd_name)(grad)
	g = tf.get_default_graph()
	with g.gradient_override_map({"PyFunc":rnd_name}):
		return tf.py_func(func,inp,tout,stateful = stateful, name=name)
	
def tf_spiky(x, name=None):
	with tf.name_scope(name,"spiky",[x]) as name:
		y = py_func(vd,[x],[tf.float32],name=name,grad=tf_d_spiky)
		return y[0]

n_classes = 10
batch_size = 128
#x = tf.placeholder('float32',[4])
y = tf.placeholder('float32',[4])
#w = tf.Variable(tf.constant(4,dtype=tf.int32,shape=[4]))
#b = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[10]))

#output = tf.nn.relu(tf.add(tf.matmul(tf_spiky(w),x),b))

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))
#optimizer = tf.train.AdamOptimizer(0.05).minimize(cost)

#correct = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct,'float'))
#hm_epochs = 2000

with tf.Session(config=config) as sess:
	x=tf.constant([0.2,0.7,1.2,1.7])	
	output = tf_spiky(x)
	sess.run(tf.global_variables_initializer())
	labels = tf.constant([1.,0.,1.,1.]).eval()
	#print(sess.run(optimizer,feed_dict={x:train,y:labels}))
	#print(sess.run(optimizer,feed_dict={x:mnist.train.images,y:mnist.train.labels}))
	print(x.eval(),labels,tf.gradients(output,[x])[0].eval())
	#print(sess.run(optimizer,feed_dict={y:labels}))
	#print(w.eval())







	
	
'''

x = tf.constant(3)
y = tf.constant(4)
z = tf.constant(5)

a = tf.multiply(x,y)

with tf.Session() as sess:
	print(sess.run(a))

'''



'''
x = tf.Variable(tf.constant([10],dtype=tf.int32,shape = [100]))

with tf.Session(config=config) as sess:
	y  = tf.multiply(tf.dtypes.cast(x,tf.float32),3.5)
	sess.run(tf.global_variables_initializer())
	grad = tf.identity(tf.gradients(y,[x]))
	print(sess.run(x))
	print(y.eval())
	print(sess.run(grad))
'''
	
	
	
'''
input = tf.Variable([3.0,2.], dtype=tf.float32)
output = tf.identity(tf.square(tf.multiply(input,6., name="Identity")))
grad = tf.identity(tf.gradients(output, input)[0])

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	print("output:",sess.run(output))
	print("without clipping:", sess.run(grad))
'''















