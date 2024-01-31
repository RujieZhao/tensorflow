

#Based on practice5_1.py
#Purpose is to create a op for find correct parameter squence by optimizer training w*x

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
	grad_weight=np.array(input*grad)
	#grad_weight = tf.to_float(tf.multiply(grad,input))
	grad_input = np.array(weight*grad)
	#grad_input = tf.to_float(tf.multiply(grad,weight))
	return grad_weight,grad_input

def new_op(x1,x2):	
	with tf.get_default_graph().gradient_override_map({"PyFunc": 'LossGradient'}):
		output = tf.py_func(np_func,inp=[x1,x2],Tout=tf.float32)
	return output

def train_linear_regression(x_data,y_data,epoch):	
	g = tf.get_default_graph()
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)

	weight = tf.Variable(tf.constant([1],dtype = tf.float32))
	output = new_op(weight,x)
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data,logits=output))
	cost = tf.reduce_mean(tf.nn.l2_loss(y-output))
	optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

	with tf.Session(graph = g,config=config) as sess:
		#weight = tf.Variable(tf.random_uniform([1],-1.,1.))
		sess.run(tf.global_variables_initializer())			
		#print(x_data.shape,y_data.shape,type(x_data))
		#z = new_op(weight,x_data)
		#print(z.eval())		
		#print("GD:",tf.gradients(new_op(weight,x_data),[weight,x_data])[0].eval(),weight.eval(),x_data)

		print("x_data_shape:{}".format(x_data.shape))
		print("y_data_shape:{}".format(y_data.shape))
		for i in range(epoch):
			_,loss=sess.run([optimizer,cost],feed_dict={x:x_data,y:y_data})
			weight = sess.run(weight)
			print("step:{},loss:{},weight:{}".format(i+1,loss,weight))

if __name__=='__main__':
		w = 0.1
		num_points=10
		#b = 0.3
		epoch = 5		
		x_data,y_data = gen_data(w,num_points)
		train_linear_regression(x_data,y_data,epoch)
		
		#plt.scatter(x_data,y_data)
		#plt.scatter(x_data,y)
		#plt.show()












