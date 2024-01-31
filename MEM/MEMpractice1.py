
'''
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
np.set_printoptions(threshold=np.inf)

a = tf.constant([[1,2],[3,4],[5,6]],tf.float32)
g=tf.Variable(tf.ones([3,2]),'float32')
b=tf.Variable(tf.constant([2,2,2,2],tf.float32,shape=[2,2]))
c=10.

def start(n):		
	with tf.Session()as sess:
		sess.run(tf.global_variables_initializer())
		#print('a:',sess.run(a))
		#print('a_shape:',a.shape)
		for i in range(0,2):
			sess.run(tf.assign(g[n,i],a[n,i]*a[n,i]))
			#s = sess.run(g)[n,:]			
		#s_b = sess.run(b)
		#print(s_b.shape)
		#print(s)[n,:], np.array(
		
		return np.array(sess.run(b)),np.array(sess.run(g)[n,:]),np.array(c)
		

def start_1(n):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.assign(g,ab))
		print('g:',sess.run(g))
		for i in range(0,4):
			sess.run(tf.assign(g[n,i],g[n,i]*g[n,i]))
			s = sess.run(g)[n,:]		
		return s,sess.run(b)

for u in range(0,1):		
	if u == 0:
		with Pool(processes=10) as p:
			#r =np.array(p.map(start,range(3),chunksize =3))
			r = p.map(start,range(3),chunksize =2)
			print('r:',r)
			#print('r_shape',r.shape)
			print('r_shape',len(r))
			r_b = r[0,0]
			print('r_b:',r_b)
			#print('r_type:',type(r))
			#print('r_b_shape:',r_b.shape)

	else:
		with Pool(processes=5) as p:
			r =np.array(p.map(start_1,range(2),chunksize =2))
			print('r:',r)
			print('r_shape:',r.shape)
			#ab = r[:,0]
			#print('ab:',ab)
			#print('ab_shape:',ab.shape) 
'''

'''
import math
import matplotlib.pyplot as plt	
import numpy as np
np.set_printoptions(threshold=np.inf)
Ry=2.3551 #11.76148
Ra= -2.421 #-12.09054
Rt=23.0499 #23.05015
Dy = 0.6959 #0.77
Da = 3.0708 #232.57604
Dt = 12.2756 #8.69255	
Points=1.
shrink=1.
#shrink=(Ra*np.exp(-30./Rt)+Ry)/(Da*np.exp(-1./(Dt*Points))+Dy)
#shrink=(Ra*np.exp(-30./Rt)+Ry)/(Da*np.exp(-0./(Dt*Points))+Dy)
#shrink= (Ra*np.exp(-30./Rt)+Ry)/((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-30./Dt)+Dy))
#shrink = shrink.astype(np.float32)
#print('shrink:',shrink)
#print('shrink_dtype',shrink.dtype)
#print('shape:',shrink.shape)
#shift= (Da*np.exp(-1./(Dt*Points))+Dy)*shrink-(Ra*np.exp(-30./Rt)+Ry)
#shift = shift.astype(np.float32)
#print('shift',shift)
shift=0.
x=np.arange(0.,(30.*Points)+1,1.)
y_1=Ra*np.exp(-x/(Rt*Points))+Ry
#y_2=Da*np.exp(-x/Dt)+Dy
#y_2=Da*np.exp(-x/(Dt*Points))+Dy
y_2=(Da*np.exp(-x/(Dt*Points))+Dy)*shrink-shift
rise_start=Ra*np.exp(-1./(Rt*Points))+Ry
print('rise_start:',rise_start)
rise_end=Ra*np.exp(-30./Rt)+Ry
print('rise_end:',rise_end)
drop_start=(Da*np.exp(-1./(Dt*Points))+Dy)*shrink-shift
print('drop_start:',drop_start)
drop_end=(Da*np.exp(-30./Dt)+Dy)*shrink-shift
print('drop_end:',drop_end)
#Rise_min=55+1e-5
#print(Rise_min)
for s in range(2):
	plt.subplot(1,2,s+1)
	if s==0:
		a= y_1		
	else:	
		a= y_2	
	plt.plot(x,a)	
plt.show()	
'''
'''
import math
import matplotlib.pyplot as plt
import numpy as np
Dy = 0.1468 #0.77
Da = 3.8811 #232.57604
Dt = 10.6836 #8.69255
Ry=3.2934 #11.76148
Ra= -3.2208 #-12.09054
Rt=3.9156 #23.05015
x = np.arange(1.,30.,1.)
#y=Ra*np.exp(-x/Rt)+Ry
y=Da*np.exp(-x/Dt)+Dy
plt.plot(x,y)
plt.show()	
'''
'''
import numpy as np
import matplotlib.pyplot as plt 
y1=[10,13,5,40,30,60,70,12,55,25] 
x1=range(1,11) 
x2=range(1,11) 
y2=[5,8,0,30,20,40,50,10,40,15] 
plt.plot(x1,y1,label='Frist line',linewidth=1,color='orange',marker='o', 
markerfacecolor='blue',markersize=5) 
plt.plot(x2,y2,label='second line') 
#plt.ylim(0,100)
plt.yticks(np.linspace(0,100,21))
plt.xticks(np.linspace(1,10,10))
plt.xlabel('Plot Number') 
plt.ylabel('Important var') 
plt.title('Interesting Graph\nCheck it out') 
plt.grid()
#plt.legend() 
plt.show()
'''
'''
import numpy as np
import tensorflow as tf
ACC= np.array([0.]*10,dtype=np.float32)
t = 5.
v=tf.add(tf.clip_by_value((t+2)*2,12,30),10)

with tf.Session() as sess:
	print('v:',sess.run(v))
#print(ACC[9])
#print('ACC:',ACC.dtype)
#print('ACC.SHAPE:',ACC.shape)	
'''







	











	
