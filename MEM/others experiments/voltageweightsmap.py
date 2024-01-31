



# for different voltages

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
#np.set_printoptions(threshold=np.inf)

weights = np.load('weights_0.55v_i(2).npy')
#io.savemat('weights.mat',{'array':weights})

print('weigts \n',weights,'\n the type is',type(weights),'\n the shape is',weights.shape)

for i in range(10):
	plt.suptitle('weights plot test', fontsize = 20)
	plt.subplot(2,5,i+1)
	#plt.title(i)
	plt.imshow(weights[:,i].reshape([28,28]),cmap=plt.get_cmap('seismic'))
	plt.axis('off')
	
	plt.savefig('055v_2.png',dpi = 900)
plt.show()
plt.close()







