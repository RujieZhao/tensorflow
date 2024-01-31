

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# 设置GPU按需增长
config = tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
''' 
# 下面是定义一个卷积层的通用方式
def conv_relu(kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    return None
def my_image_filter():
    # 按照下面的方式定义卷积层，非常直观，而且富有层次感
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu([5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        relu2 = conv_relu([5, 5, 32, 32], [32])
    return None
with tf.variable_scope("image_filters", reuse = tf.AUTO_REUSE):
#with tf.variable_scope("image_filters") as scope:
    # 下面我们两次调用 my_image_filter 函数，但是由于引入了 变量共享机制
    # 可以看到我们只是创建了一遍网络结构。
    result1 = my_image_filter()
    print('R1:',result1)
    #scope.reuse_variables()
    result2 = my_image_filter()

# 看看下面，完美地实现了变量共享！！！
vs = tf.trainable_variables()
print('There are %d train_able_variables in the Graph: ' % len(vs))
for v in vs:
    print(v)



sess.run(tf.global_variables_initializer())
print('vs:',sess.run(vs))

print('vs_shape:',np.array(sess.run(vs)).shape)
'''
'''
#在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope("foo"):
    #创建一个常量为1的v
    v= tf.get_variable('v',[1],initializer = tf.constant_initializer(1.0))
    v3 = tf.get_variable('z',[1],initializer = tf.constant_initializer(2.0))
#因为在foo空间已经创建v的变量，所以下面的代码会报错
#with tf.variable_scope("foo"）:
#   v= tf.get_variable('v',[1])
#在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable的函数将直接获取已声明的变量
#且调用with tf.variable_scope("foo"）必须是定义的foo空间，而不能是with tf.variable_scope(""）未命名或者其他空间。
with tf.variable_scope("foo",reuse =True ):
    v1= tf.get_variable('v',[1],initializer = tf.constant_initializer(2.))#  不写[1]也可以
    v2 = tf.get_variable('z')
    #v3 = tf.get_variable('z')
    
    print(v1==v) #输出为True，代表v1与v是相同的变量
sess.run(tf.global_variables_initializer())
print(sess.run(v))
print(sess.run(v1))
print(sess.run(v2))
#print(sess.run(v3))
'''

x = tf.Variable(tf.constant([1],dtype = tf.int32,shape = [1]))
x1 = tf.Variable(tf.constant([1],dtype = tf.int32,shape = [101]))
#y = tf.Variable(tf.constant([1],dtype = tf.int32))
xx= np.array(range(0,101,1))
yy = np.array([0.]*101,dtype=np.float32)

def func(x):
	y = tf.multiply(tf.dtypes.cast(x,tf.float32),3.5)
	return y[0]
	

with tf.Session(config = config) as sess:
	y = tf.identity(tf.multiply(tf.dtypes.cast(x,tf.float32),3.5))
	sess.run(tf.global_variables_initializer())
	Y = sess.run(y,feed_dict={x:[1.9]})
	X = sess.run(tf.assign(x,[2]))
	print(type(X))
	print('Y:',Y)
	print(sess.run(x).dtype)
	print(xx.shape,'\n',yy.shape)
	print(type(sess.run(x1)))
	x2 = sess.run(x1)
	print(type(x2))
	print(sess.run(func(x2)))
	print(tf.identity(tf.gradients(y,x))[0].eval())
	for i in range(0,101):
		yy[i] = sess.run(y,feed_dict={x:[xx[i]]})
'''
plt.plot(xx,yy)
plt.show()
plt.close()
'''
		
'''
	xx = range(0,20,0.1)
	yy = tf.placeholder(dtype = tf.float32, shape = [200])
	sess.run(tf.global_variables_initializer())
	for i in range(20):
		yy[i] = sess.run(y,feed_dict={x:xx[i]})
	plt.plot(xx, yy)
	plt.show()
	plt.close()
'''	
	
	
	
	
	
	

















