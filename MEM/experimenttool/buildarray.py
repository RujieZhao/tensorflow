

#Build a multiple dimensions array

import numpy as np

# ============one dimension===============

#a = np.arange(20)
#a = np.array([0 for i in range(4)])
#a = np.array([0]*5)


# ============two dimension===============

#a = np.array([[0,0],[0,0],[0,0]])
#a = np.array([[0 for i in range(2)] for i in range(3)])
#a = np.array([[0,0]]*3)

# ============three dimension===============

#a = np.array([[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]])
#a = np.array([[[0 for i in range(50)] for i in range(50)]for i in range(100)])
a = np.array([[[0,0]]*3]*2)

print(a)
print(a.shape)













