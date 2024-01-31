



import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
#config = tf.ConfigProto(log_device_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
mnist = input_data.read_data_sets("/home/rujie/tensorflow/MNIST_data/",one_hot=True)

def gen_data(w,num_points):
	vectors_set=[]
	x_data = []
	y_data = []
	for i in range(num_points):
		x1 = np.random.normal(0.,0.55)
		y1 = x1*w+np.random.normal(0.,0.03)
		vectors_set.append([x1,y1])
		x_data=np.array([v[0] for v in vectors_set],dtype=np.float32)
		y_data=np.array([v[1] for v in vectors_set],dtype=np.float32)
		#print('x_data_shape',x_data.shape,type(x_data),x_data.dtype)
		#plt.scatter(x_data,y_data,c='b')
	#plt.show()
	return x_data,y_data

def np_func (weight,x_data):
	y = x_data*weight
	return  np.array(y,dtype=np.float32)
np_vfunc = np.vectorize(np_func)

@tf.RegisterGradient("LossGradient")
def np_func_grad(op, grad):
	weight = op.inputs[0]	
	input = op.inputs[1]	
	#grad_weight=input*grad
	#grad_weight = tf.to_float(tf.multiply(grad,input))
	#grad_input = wei*grad
	#grad_input = tf.to_float(tf.multiply(grad,weight))
	#return grad_weight,grad_input
	return grad*2,grad*3
def new_op(x1,x2):	
	with tf.get_default_graph().gradient_override_map({"PyFunc": 'LossGradient'}):
		output = tf.py_func(np_func,inp=[x1,x2],Tout=tf.float32)
	return output

def train_linear_regression(x_data,y_data):	
	g = tf.get_default_graph()
	weight = tf.constant([0.1],dtype = tf.float32)
	z = new_op(weight,x_data)
	GD = tf.gradients(z,[weight,x_data])
	with tf.Session(graph=g,config=config) as sess:
		#weight = tf.Variable(tf.random_uniform([1],-1.,1.))
		sess.run(tf.global_variables_initializer())			
		print(x_data.shape,y_data.shape,type(x_data))
		
		print("Z:",z.eval())		
		print("GD:",GD,weight.eval(),x_data)

'''
	print("x_data_shape:{}".format(x_data.shape))
	print("y_data_shape:{}".format(y_data.shape))
	weight = tf.Variable(tf.random_uniform([1],-1.,1.))
	output = new_op(weight,x_data)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data,logits=output))
	#cost = tf.reduce_mean(tf.nn.l2_loss(y_data-output)+tf.multiply(0.1,tf.nn.l2_loss(w)))
	optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(epoch):
			_,loss=sess.run([optimizer,cost])
			weight = sess.run(weight)
			print("step:{},loss:{},weight:{}".format(i+1,loss,weight))
'''
if __name__=='__main__':
		w = 0.1
		num_points=1
		#b = 0.3		
		x_data,y_data = gen_data(w,num_points)
		train_linear_regression(x_data,y_data)
		
		#plt.scatter(x_data,y_data)
		#plt.scatter(x_data,y)
		#plt.show()


'''
def np_spiky(x):
	x = np.int(x)
	#print('x from np_spiky:',x)
	y = math.pow(x,2)*1.5
	#y = x
	#print('y from np_spiky:',y)
	return y
v = np.vectorize(np_spiky)
vd = lambda x:np.float32(v(x))
print(np_spiky(2.5),type(np_spiky(2.5)),type(v(2.5)),vd(2.5))

def tf_d_spiky(op,grad):
	x = tf.cast(op.inputs[0],tf.int32)
	#print("XXXXXXXXXXX:",x)
	n_gr = tf.to_float(tf.multiply(x,3))
	#print("YYYYYYYYYY",n_gr)
	return n_gr*grad

#@tf.RegisterGradient("LossGradient")
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
#a = tf.Variable(tf.random_normal([4],mean = 2,stddev=0.2,dtype = tf.float32))
a = tf.Variable(tf.random_uniform([4],0,8,dtype = tf.dtypes.int32))
y=tf.constant([0.2,0.7,1.2,1.7])
y_1 = tf.cast (y,tf.int32)
y_2 = tf.cast (y_1,tf.float32)
#y = tf.constant([3,4,5,6],dtype=tf.float32)
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='W')

#output = tf.nn.relu(tf.add(tf.matmul(tf_spiky(w),x),b))
output = tf_spiky(y)
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))
#optimizer = tf.train.AdamOptimizer(0.05).minimize(cost)

#correct = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct,'float'))
#hm_epochs = 2000

with tf.Session(config=config) as sess:
		
	sess.run(tf.global_variables_initializer())
	print(y_2.eval())
	print('w:',w.eval())
	print('a_original:',a.eval())
	print('output:',output.eval())
	print('GD:',tf.gradients(output,[y])[0].eval())
	#labels = tf.constant([1.,0.,1.,1.]).eval()
	#print("COST:",sess.run(cost,feed_dict={y:labels}))
	#print(sess.run(optimizer,feed_dict={y:labels}))
	#print(a.eval(),output.eval(),tf.gradients(output,[a])[0].eval())
	#print(sess.run(optimizer,feed_dict={y:labels}))
	#print(w.eval())
'''






	
	
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















