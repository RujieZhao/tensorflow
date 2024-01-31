




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
	diff_d[i] = y_2(i)-y_2(i+1) 
	sum_d = sum_d+diff_d[i]
M_r = sum_r/len(diff_r)
M_d = sum_d/len(diff_d) 
print('diff_r:',diff_r.shape,' diff_d:',diff_d.shape)
print('Rise Mean:',M_r,' Drop Mean:',M_d)

diff_r_max = np.max(diff_r)
diff_r_max_index = np.where(diff_r == diff_r_max)
print('diff_r_max:',diff_r_max,' diff_r_max_index:',diff_r_max_index)

diff_d_max = np.max(diff_d)
diff_d_max_index = np.where(diff_d == diff_d_max)
print('diff_d_max:',diff_d_max,' diff_d_max_index:',diff_d_max_index)

diff_r_min = np.min(diff_r)
diff_r_min_index = np.where(diff_r == diff_r_min)
print('diff_r_min:',diff_r_min,'diff_r_min_index:',diff_r_min_index)

diff_d_min = np.min(diff_d)
diff_d_min_index = np.where(diff_d == diff_d_min)
print('diff_d_min:',diff_d_min,'diff_d_min_index:',diff_d_min_index)


# drop = np.vectorize(y_2)
# print('rise_start:',y_1(0.))
# print('rise_end:',rise(163.))
# print('drop_start:',drop(1.))
# print('drop_end:',drop(161.))

# plt.plot(x1,y_1(x1))
# plt.plot(x2,y_2(x2))
# plt.show()





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
Points=3

diff_r =np.array([0. for _ in range(30*Points)]) 
diff_d =np.array( [0. for _ in range(30*Points)])

sum_r = 0
sum_d = 0

x1=np.arange(0,(30*Points)+1,1)
x2=np.arange(0,(30*Points)+1,1)

shrink1 = 3.5/(Ra*(np.exp(-30./Rt)-1.))
shrink2 = 3.5/(Da*(1-np.exp(-30./Dt)))
shift1 = (Ra+Ry)*shrink1
shift2 = (Da*np.exp(-30./Dt)+Dy)*shrink2

#shrink1 = 1.
#shrink2 = 1.
#shift1 = 0.
#shift2 = 0.

def y_1(x):
	return (Ra*np.exp(-x/(Rt*Points))+Ry)*shrink1-shift1
def y_2(x):
	return (Da*np.exp(-x/(Dt*Points))+Dy)*shrink2-shift2

for r in range(0,30*Points):
	diff_r[r] = y_1(r+1)-y_1(r)
	sum_r = sum_r+diff_r[r]
for i in range(0,30*Points):
	diff_d[i] = y_2(i)-y_2(i+1) 
	sum_d = sum_d+diff_d[i]
M_r = sum_r/len(diff_r)
M_d = sum_d/len(diff_d) 
print('diff_r:',diff_r.shape,'\n diff_d:',diff_d.shape)
print('Rise Mean:',M_r,'\n Drop Mean:',M_d)

diff_r_max = np.max(diff_r)
diff_r_max_index = np.where(diff_r == diff_r_max)
print('diff_r_max:',diff_r_max,' diff_r_max_index:',diff_r_max_index)

diff_d_max = np.max(diff_d)
diff_d_max_index = np.where(diff_d == diff_d_max)
print('diff_d_max:',diff_d_max,' diff_d_max_index:',diff_d_max_index)

diff_r_min = np.min(diff_r)
diff_r_min_index = np.where(diff_r == diff_r_min)
print('diff_r_min:',diff_r_min,'diff_r_min_index:',diff_r_min_index)

diff_d_min = np.min(diff_d)
diff_d_min_index = np.where(diff_d == diff_d_min)
print('diff_d_min:',diff_d_min,'diff_d_min_index:',diff_d_min_index)

# print('rise_start:',y_1(0.))
# print('rise_end:',y_1(30.))
# print('drop_start:',y_2(0.))
# print('drop_end:',y_2(30.))

# plt.plot(x1,y_1(x1))
# plt.plot(x2,y_2(x2))
# plt.show()
'''



