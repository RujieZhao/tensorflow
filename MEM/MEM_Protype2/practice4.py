



# GPU parallel


import tensorflow as tf
import numpy as np
a= tf.ones([2,4])
b = tf.zeros([2,4])

def make_parallel(num,**kwargs):
	in_splits = {}
	for k,v in kwargs.items():
		in_splits [k] = tf.split(v,num)
		print('k:',k,'.\nv:',v)
	return in_splits

with tf.Session() as sess:
		
	p = make_parallel(2,a=sess.run(a),b=sess.run(b))
	split_a = tf.split(a,2)
	print(np.array(sess.run(split_a))[1])
	print(p)




'''
with tf.device('/gpu:0'):
	a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3], name='a')
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
	c = tf.matmul(a, b)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #sess.run(tf.global_variables_initializer())
    print (sess.run(c))

gpus = tf.config.experimental.list_physical_devices('GPU')
with tf.Session() as sess:
	for gpu in gpus:
		print("Name:", gpu.name, "  Type:", gpu.device_type)
'''

'''
allow_soft_placement=True, 
#with tf.device(tf.DeviceSpec(device_type="GPU", device_index=2)):
a = tf.random_uniform([1000, 100])
b = tf.random_uniform([1000, 100])
c = a + b

with tf.Session() as sess:
	print(sess.run(c))
'''


