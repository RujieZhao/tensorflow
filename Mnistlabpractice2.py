
'''
import numpy as np
data_w=np.load('w_new.npy')
print('data_w:',data_w,'\n the type is:',type(data_w))
'''

'''
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32',[None,10])
w = tf.Variable(tf.random.normal([784,10],dtype=tf.float32))
b = tf.Variable(tf.constant(1,dtype=tf.float32,shape=[10]))
output = tf.nn.relu(tf.matmul(x,w)+b)
		
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= output))		
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
hm_epochs = 2000

with tf.Session() as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	epoch_loss=0
	for i in range(1,hm_epochs+1):		
		_,c=sess.run([optimizer,cost],feed_dict={x:mnist.train.images, y:mnist.train.labels})
		epoch_loss += c
		if i %1000 == 0:
			print('Epoch:',i,'completed out of', hm_epochs, 'loss:', epoch_loss)
			print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
		#if i==hm_epochs:
			#weight=sess.run(w)
			#np.save('w_new.npy',weight)
'''
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float32', [None, 784])
y = tf.placeholder('float32',[None,10])
w = tf.Variable(tf.random.normal([784,10],dtype=tf.float32))
b = tf.Variable(tf.constant(1,dtype=tf.float32,shape=[10]))
output = tf.nn.relu(tf.matmul(x,w)+b)
		
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels= y, logits= output))		
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

correct = tf.equal(tf.argmax(output,1), tf.argmax(y,1)) 
accuracy = tf.reduce_mean(tf.cast(correct,'float'))
hm_epochs = 20

with tf.Session() as sess:
	#sess.run(tf.initialize_all_variables())
	sess.run(tf.global_variables_initializer())
	epoch_loss=0
	for i in range(1,hm_epochs+1):		
		_,c=sess.run([optimizer,cost],feed_dict={x:mnist.train.images, y:mnist.train.labels})
		epoch_loss += c
	print(sess.run(tf.clip_by_value(w,0.,1.)))		
'''
		
'''	
		if i %1000 == 0:
			print('Epoch:',i,'completed out of', hm_epochs, 'loss:', epoch_loss)
			print('Accuracy:', accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))			
		
			for s in range(10):
				weight=sess.run(w)[:,s]
				plt.suptitle('thread test',fontsize = 20)
				plt.subplot(2,5,s+1)
				plt.title(s)
				plt.imshow(weight.reshape([28,28]),cmap=plt.get_cmap('seismic'))
				plt.axis('off')
			#plt.savefig('w_i('+str(i)+').png',dpi = 400)		
			plt.show()
			plt.close()
'''
'''
import math
import matplotlib.pyplot as plt	
import numpy as np
np.set_printoptions(threshold=np.inf)

Ry=9.253
Ra=-9.9504
Rt=79.6095
Dy =-195.218
Da =200.9147
Dt =5786

Points=1.
shrink=1.
shrink2=1.


#shrink = ((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-30./Dt)+Dy))/((Ra*np.exp(-30./Rt)+Ry)-(Ra*np.exp(-0./Rt)+Ry))
#shrink = 2./((Ra*np.exp(-18./Rt)+Ry)-(Ra*np.exp(-0./Rt)+Ry))
#shrink2= 2./((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-25./Dt)+Dy))
#shrink2 =((Ra*np.exp(-25./Rt)+Ry)-(Ra*np.exp(-0./Rt)+Ry))/((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-35./Dt)+Dy))
#print('shrink2:',shrink2)
#shrink=(Ra*np.exp(-30./Rt)+Ry)/(Da*np.exp(-1./(Dt*Points))+Dy)
#shrink=(Ra*np.exp(-30./Rt)+Ry)/(Da*np.exp(-0./(Dt*Points))+Dy)
#shrink= (Ra*np.exp(-30./Rt)+Ry)/((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-30./Dt)+Dy))
#shrink = shrink.astype(np.float32)
#print('shrink_dtype',shrink.dtype)
#print('shape:',shrink.shape)
#shift2 = (Ra*np.exp(-25./Rt)+Ry)*shrink-(Da*np.exp(-1./(Dt*Points))+Dy)
#shift = shift.astype(np.float32)
#print('shift',shift)
#shift = (Ra*np.exp(-0./Rt)+Ry)*shrink
#shift2 = (Da*np.exp(-161./Dt)+Dy)*shrink2 
#print('shift2:',shift2)
#shift = (Ra*np.exp(-0./Rt)+Ry)*shrink-(Da*np.exp(-30./(Dt*Points))+Dy)
shift=0.
shift2=0.

x1=np.arange(0.,(163.*Points)+1,1.)
x2=np.arange(0.,(161.*Points)+1,1.)
#y_1=Ra*np.exp(-x1/(Rt*Points))+Ry
y_1 = (Ra*np.exp(-x1/(Rt*Points))+Ry)*shrink-shift
#y_2=Da*np.exp(-x/Dt)+Dy
y_2=(Da*np.exp(-x2/(Dt*Points))+Dy)*shrink2-shift2
#y_2=(Da*np.exp(-x2/(Dt*Points))+Dy)*shrink-shift
rise_start=(Ra*np.exp(-0./(Rt*Points))+Ry)*shrink-shift
print('rise_start:',rise_start)
rise_end=(Ra*np.exp(-163./Rt)+Ry)*shrink-shift
print('rise_end:',rise_end)
#drop_start=(Da*np.exp(-1./(Dt*Points))+Dy)*shrink-shift
drop_start=(Da*np.exp(-1./(Dt*Points))+Dy)*shrink2-shift2
print('drop_start:',drop_start)
#drop_end=(Da*np.exp(-35./Dt)+Dy)*shrink-shift
drop_end=(Da*np.exp(-161./Dt)+Dy)*shrink2-shift2
print('drop_end:',drop_end)
#Rise_min=55+1e-5
#print(Rise_min)
for s in range(1):
	plt.subplot(1,1,s+1) #(row, column)
	if s==0:
		a= y_1		
		
	#else:	
		b= y_2	
	plt.plot(x1,a)
	plt.plot(x2,b)	
plt.show()
'''

import math
import matplotlib.pyplot as plt	
import numpy as np
Ry=0.0024   #9.253
Ra=-6068e-04    #-9.9504
Rt= 29.4191     #79.6095
Dy = 3.01e-4       #-195.218
Da = 0.0019       #200.9147
Dt = 109.12 

Points=1.
shrink =2./((Ra*np.exp(-27./(Rt))+Ry)-(Ra*np.exp(-0./(Rt))+Ry))
shrink2 = 2./((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-27./(Dt))+Dy))
#shrink=1.
#shrink= (Ra*np.exp(-30./Rt)+Ry)/((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-30./Dt)+Dy))
#shrink = shrink.astype(np.float32)
#print('shrink:',shrink)
#print('shrink_dtype',shrink.dtype)
#print('shape:',shrink.shape)
#shift= (Da*np.exp(-1./(Dt*Points))+Dy)*shrink-(Ra*np.exp(-30./Rt)+Ry)
#shift = shift.astype(np.float32)
#print('shift',shift)
#shift=0.
shift2 = (Da*np.exp(-27./(Dt))+Dy)*shrink2 
shift = (Ra*np.exp(-0./(Rt))+Ry)*shrink
# a_r = (Ra*np.exp(-30./Rt)-Ra*np.exp(-1./(Points*Rt)))/(30.*Points-1)
# b_r = Ra*np.exp(-1./(Points*Rt))+Ry-a_r
# a_d = (Da*np.exp(-30./Dt)-Da*np.exp(-1./(Points*Dt)))/(30.*Points-1)
# b_d = Da*np.exp(-1./(Points*Dt))+Dy-a_d

# print('a_r:',a_r,'\nb_r:',b_r,'\na_d:',a_d,'\nb_d',b_d)

x=np.arange(0.,(27.*Points)+1,1.)
# y_1 = a_r*x+b_r
# y_2 = (a_d*x+b_d)*shrink-shift
Y_1 = (Ra*np.exp(-x/(Rt*Points))+Ry)*shrink-shift
Y_2 = (Da*np.exp(-x/(Dt*Points))+Dy)*shrink2-shift2
# rise_start = a_r*1.+b_r
rise_start = (Ra*np.exp(-0./(Rt*Points))+Ry)*shrink-shift
print('rise_start:',rise_start)
# rise_end = a_r*(30.*Points)+b_r
rise_end = (Ra*np.exp(-27./(Rt*Points))+Ry)*shrink-shift
print('rise_end:',rise_end)
# drop_start = a_d*1+b_d
drop_start = (Da*np.exp(-1./(Dt*Points))+Dy)*shrink2-shift2
print('drop_start:',drop_start)
# drop_end = a_d*(30.*Points)+b_d
drop_end = (Da*np.exp(-27./(Dt*Points))+Dy)*shrink2-shift2
print('drop_end:',drop_end)

for s in range (2):
	plt.subplot(1,2,s+1)
	if s==0:
		# a = y_1
		b = Y_1
	else:
		# a = y_2
		b = Y_2
	# plt.plot(x,a)
	plt.plot(x,b)
	plt.title(s)
plt.show()

'''
import math
import matplotlib.pyplot as plt	
import numpy as np

Ra=3.89E-04
Rb=0.3687
Da=-5.80E-04
Db=0.7611
print(Ra)
shrink1 = 2.5/(Ra*582.)
shrink2= 2.5/(Da*1.-Da*491.)
shift1 = Rb*shrink1
shift2 = (Da*491.+Db)*shrink2

x1=np.arange(0,582+1,1)
x2=np.arange(0,491+1,1)

y_1 = (Ra*x1+Rb)*shrink1-shift1
y_2 = (Da*x2+Db)*shrink2-shift2

rise_start=(Rb)*shrink1-shift1
print('rise_start_orginal:',Rb)
print('rise_start:',rise_start)
rise_end=(Ra*582.+Rb)*shrink1-shift1
print('rise_end_orginal:',Ra*582.+Rb)
print('rise_end:',rise_end)
drop_start=(Da*1.+Db)*shrink2-shift2
print('drop_start_orginal:',Da*1.+Db)
print('drop_start:',drop_start)
drop_end= (Da*491.+Db)*shrink2-shift2
print('drop_end_orginal:',Da*491.+Db)
print('drop_end:',drop_end)

plt.subplot(1,2,1)
plt.plot(x1,y_1)
plt.subplot(1,2,2)
plt.plot(x2,y_2)
plt.show()
'''
'''
import math
import matplotlib.pyplot as plt	
import numpy as np
np.set_printoptions(threshold=np.inf)

Ry=9.253
Ra=-9.9504
Rt=79.6095
Dy =-195.218
Da =200.9147
Dt =5786
Points=1.

x1=np.arange(0.,(163.*Points)+1,1.)
x2=np.arange(1.,(161.*Points)+1,1.)
Rk = np.float32((Ra*np.exp(-163./Rt)-Ra)/(163.*Points))
Rb = np.float32(Ra+Ry)
#print(type(Rb))
#print(type(Rk))
#print('Rk: %a; Rb: %a' %(Rk, Rb))
Dk = np.float32((Da*np.exp(-161./Dt)-Da*np.exp(-1./Dt))/(161.*Points-1.))
Db = np.float32(Da*np.exp(-1./Dt)+Dy-Dk)

#shrink1 = 3.5/(Rk*(163.*Points))
#shrink2 = 3.5/(Dk*1.-Dk*(161.*Points))
#shift1 = Rb*shrink1
#shift2 = (Dk*(161.*Points)+Db)*shrink2

shrink1 = 1.
shrink2 = 1.
shift1 = 0.
shift2 = 0.


y_1 = (Rk*x1+Rb)*shrink1-shift1
y_2 = (Dk*x2+Db)*shrink2-shift2
print('Ra_L:',np.float32(Rb*shrink1-shift1))
print('Rb_L:',np.float32((Rk*(163*Points)+Rb)*shrink1-shift1))
print('Da_L:',np.float32((Dk+Db)*shrink2-shift2))
print('Db_L:',np.float32((Dk*(161*Points)+Db)*shrink2-shift2))

print('Ra:',np.float32((Ra+Ry)*shrink1-shift1))
print('Rb:',np.float32((Ra*np.exp(-163./Rt)+Ry)*shrink1-shift1))
print('Da:',np.float32((Da*np.exp(-1./Dt)+Dy)*shrink2-shift2))
print('Db:',np.float32((Da*np.exp(-161./Dt)+Dy)*shrink2-shift2))

plt.plot(x1,y_1)
plt.plot(x2,y_2)
plt.show()
'''
'''
#y_1=Ra*np.exp(-x1/(Rt*Points))+Ry
y_1 = (Ra*np.exp(-x1/(Rt*Points))+Ry)*shrink1-shift1
#y_2=Da*np.exp(-x/Dt)+Dy
y_2=(Da*np.exp(-x2/(Dt*Points))+Dy)*shrink2-shift2
#y_2=(Da*np.exp(-x2/(Dt*Points))+Dy)*shrink-shift
rise_start=(Ra*np.exp(-0./(Rt*Points))+Ry)*shrink1-shift1
print('rise_start:',rise_start)
rise_end=(Ra*np.exp(-163./Rt)+Ry)*shrink1-shift1
print('rise_end:',rise_end)
#drop_start=(Da*np.exp(-1./(Dt*Points))+Dy)*shrink-shift
drop_start=(Da*np.exp(-1./(Dt*Points))+Dy)*shrink2-shift2
print('drop_start:',drop_start)
#drop_end=(Da*np.exp(-35./Dt)+Dy)*shrink-shift
drop_end=(Da*np.exp(-161./Dt)+Dy)*shrink2-shift2
print('drop_end:',drop_end)
plt.plot(x1,y_1)
plt.plot(x2,y_2)
plt.show()

'''
'''
import math
import matplotlib.pyplot as plt	
import numpy as np
np.set_printoptions(threshold=np.inf)

Ry=9.253
Ra=-9.9504
Rt=79.6095
Dy =-195.218
Da =200.9147
Dt =5786
Points=1.

diff_r =np.array([0. for _ in range(163)]) 
diff_d =np.array( [0. for _ in range(161)])

sum_r = 0
sum_d = 0

x1=np.arange(0.,(163.*Points)+1,1.)
x2=np.arange(0.,(161.*Points)+1,1.)

shrink1 = 3.5/(Ra*(np.exp(-163./Rt)-1.))
shrink2 = 3.5/(Da*(np.exp(-1./Dt)-np.exp(-161./Dt)))
shift1 = (Ra+Ry)*shrink1
shift2 = (Da*np.exp(-161./Dt)+Dy)*shrink2

#shrink1 = 1.
#shrink2 = 1.
#shift1 = 0.
#shift2 = 0.

def y_1(x):
	return (Ra*np.exp(-x/Rt)+Ry)*shrink1-shift1
def y_2(x):
	return (Da*np.exp(-x/Dt)+Dy)*shrink2-shift2

for r in range(0,163):
	diff_r[r] = y_1(r+1)-y_1(r)
	sum_r = sum_r+diff_r[r]
for i in range(0,161):
	diff_d[i] = y_2(i+1)-y_2(i) 
	sum_d = sum_d+diff_d[i]
M_r = sum_r/len(diff_r)
M_d = sum_d/len(diff_d) 
print('diff_r:',diff_r.shape,'\n diff_d:',diff_d.shape)
print('Rise Mean:',M_r,'\n Drop Mean:',M_d)

# drop = np.vectorize(y_2)
# print('rise_start:',y_1(0.))
# print('rise_end:',rise(163.))
# print('drop_start:',drop(1.))
# print('drop_end:',drop(161.))

plt.plot(x1,y_1(x1))
plt.plot(x2,y_2(x2))
plt.show()
'''

'''
import math
import matplotlib.pyplot as plt	
import numpy as np

np.set_printoptions(threshold=np.inf)
p_r = 30.
p_d = 30.
ratio = 3.5

k_r = ratio/p_r
k_d = ratio/(1-p_d)
b_d = ratio-k_d


x1=np.arange(0.,p_r+1,1.)
x2=np.arange(1.,p_d+1,1.)

def y_1(x):
	return np.float32(k_r*x)

def y_2(x):
	return np.float32(k_d*x+b_d)

def x_1(y):
	return np.float32(y/k_r)

def x_2(y):
	return np.float32((y-b_d)/k_d)

#rise = np.vectorize(y_1)
#drop = np.vectorize(y_2)
print('rise_start:',y_1(0.))
print('rise_end:',y_1(p_r))
print('drop_start:',y_2(1.))
print('drop_end:',y_2(p_d ))

print('s_r:',x_1(1.2068965))
print('s_d:',x_2(1.2068965))

# plt.plot(x1,y_1(x1))
# plt.plot(x2,y_2(x2))
# plt.show()
'''
'''
import math
import matplotlib.pyplot as plt	
import numpy as np

np.set_printoptions(threshold=np.inf)
p_r = 30.
p_d = 30.
ratio = 3.5

k_r = ratio/p_r
k_d = -ratio/p_d
b_d = ratio

x1=np.arange(0.,p_r+1,1.)
x2=np.arange(1.,p_d+1,1.)

def y_1(x):
	return np.float32(k_r*x)

def y_2(x):
	return np.float32(k_d*x+b_d)
#rise = np.vectorize(y_1)
#drop = np.vectorize(y_2)
print('rise_start:',y_1(0.))
print('rise_end:',y_1(p_r))
print('drop_start:',y_2(0.))
print('drop_end:',y_2(p_d ))

def x_1(y):
	return np.float32(y/k_r)

def x_2(y):
	return np.float32((y-b_d)/k_d)

print('s_r:',x_1(0.9333334))
print('s_d:',x_2(0.9333334))

# plt.plot(x1,y_1(x1))
# plt.plot(x2,y_2(x2))
# plt.show()
'''



