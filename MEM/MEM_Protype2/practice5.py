



from tensorflow.python.framework import ops
import tensorflow as tf
import numpy as np

def spiky(x):
    r = x % 1
    if r <= 0.5:
        return r
    else:
        return 0
#print("xxxxxxxxxxxxxxx:",spiky(1.2))
np_spiky = np.vectorize(spiky)
print(np_spiky(1.7))
def d_spiky(x):
    r = x % 1
    if r <= 0.5:
        return 1
    else:
        return 0
np_d_spiky = np.vectorize(d_spiky)

np_d_spiky_32=lambda x:np.float32(np_d_spiky(x))
print("np_d_spiky_32:",np_d_spiky_32(1.7))  
   
def tf_d_spiky(x,name=None):
    with tf.name_scope(name, "d_spiky", [x]) as name:
        y = tf.py_func(np_d_spiky_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y

def spikygrad(op, grad):
	x = op.inputs[0]
	#print('op:',op)
	n_gr = tf_d_spiky(x)
	return grad * n_gr  
    
def spikygrad2(op, grad):
    x = op.inputs[0]
    r = tf.mod(x,1)
    n_gr = tf.to_float(tf.less_equal(r, 0.5))
    return grad * n_gr  

np_spiky_32 = lambda x: np_spiky(x).astype(np.float32)
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_spiky(x, name=None):

    with tf.name_scope(name, "spiky", [x]) as name:
        y = py_func(np_spiky_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=spikygrad)  # <-- here's the call to the gradient
        return y[0]

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    x = tf.constant([0.2,0.7,1.2,1.7])
    y = tf_spiky(x)
    tf.initialize_all_variables().run()

    print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())      






'''
a = tf.constant([[[1],[2]],[[3],[4]]])
b = tf.constant([1,2])
w = tf.get_variable("W",shape = [5])
check_a = tf.assert_rank(a,3)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
	sess.run(tf.global_variables_initializer())
	
	with tf.control_dependencies([check_a]):
	
	#print(sess.run(check_a))
	#print(sess.run(a).ndim,'\n',sess.run(b).ndim)

		weights = sess.run(w)
		
		print(sess.run(tf.nn.softmax(weights)))

		print(np.nextafter(np.float32(0), np.float32(1)))  # prints 1.4013e-45
		print(np.finfo(np.float32).max)  # print 3.40282e+38
'''
		
		
		
		
		
