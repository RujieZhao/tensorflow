


import math
import matplotlib.pyplot as plt	
import numpy as np
np.set_printoptions(threshold=np.inf)

Ry=2.8269
Ra=-3.3702
Rt=10.557
Dy =0.112
Da =1.7997
Dt =2.597

Points=1.
#shrink=1.
#shrink2=1.


shrink = 2./((Ra*np.exp(-27./Rt)+Ry)-(Ra*np.exp(-0./Rt)+Ry))
shrink2= 2./((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-34./Dt)+Dy))
#shrink2 =((Ra*np.exp(-25./Rt)+Ry)-(Ra*np.exp(-0./Rt)+Ry))/((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-35./Dt)+Dy))
#print('shrink',shrink)
#shrink=(Ra*np.exp(-30./Rt)+Ry)/(Da*np.exp(-1./(Dt*Points))+Dy)
#shrink=(Ra*np.exp(-30./Rt)+Ry)/(Da*np.exp(-0./(Dt*Points))+Dy)
#shrink= (Ra*np.exp(-30./Rt)+Ry)/((Da*np.exp(-1./(Dt*Points))+Dy)-(Da*np.exp(-30./Dt)+Dy))
#shrink = shrink.astype(np.float32)
#print('shrink:',shrink)
#print('shrink_dtype',shrink.dtype)
#print('shape:',shrink.shape)
#shift2 = (Ra*np.exp(-25./Rt)+Ry)*shrink-(Da*np.exp(-1./(Dt*Points))+Dy)
#shift = shift.astype(np.float32)
#print('shift',shift)
shift = (Ra*np.exp(-0./Rt)+Ry)*shrink
shift2 = (Da*np.exp(-34./Dt)+Dy)*shrink2 
#shift2=0.
#shift = (Ra*np.exp(-0./Rt)+Ry)*shrink-((Da*np.exp(-27./(Dt*Points))+Dy)*shrink2-shift2)
#shift=0.


x1=np.arange(0.,(27.*Points)+1,1.)
x2=np.arange(0.,(34.*Points)+1,1.)
#y_1=Ra*np.exp(-x1/(Rt*Points))+Ry
y_1 = (Ra*np.exp(-x1/(Rt*Points))+Ry)*shrink-shift
#y_2=Da*np.exp(-x/Dt)+Dy
y_2=(Da*np.exp(-x2/(Dt*Points))+Dy)*shrink2-shift2
#y_2=(Da*np.exp(-x2/(Dt*Points))+Dy)*shrink-shift
rise_start=(Ra*np.exp(-0./(Rt*Points))+Ry)*shrink-shift
print('rise_start:',rise_start)
rise_end=(Ra*np.exp(-27./Rt)+Ry)*shrink-shift
print('rise_end:',rise_end)
#drop_start=(Da*np.exp(-1./(Dt*Points))+Dy)*shrink-shift
drop_start=(Da*np.exp(-1./(Dt*Points))+Dy)*shrink2-shift2
print('drop_start:',drop_start)
#drop_end=(Da*np.exp(-35./Dt)+Dy)*shrink-shift
drop_end=(Da*np.exp(-34./Dt)+Dy)*shrink2-shift2
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




