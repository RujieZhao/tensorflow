


import matplotlib.pyplot as plt
import numpy as np
from scipy import io
#np.set_printoptions(threshold=np.inf)

weights = np.load('w_new.npy')
#io.savemat('weights.mat',{'array':weights})

print('weigts \n',weights,'\n the type is',type(weights),'\n the shape is',weights.shape)

for i in range(10):
	plt.suptitle('weights plot test', fontsize = 20)
	plt.subplot(2,5,i+1)
	#plt.title(i)
	plt.imshow(weights[:,i].reshape([28,28]),cmap=plt.get_cmap('seismic'))
	plt.axis('off')
	#plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=None, hspace=None)
	#plt.savefig('weights.png',dpi = 900)
plt.show()
plt.close()




















