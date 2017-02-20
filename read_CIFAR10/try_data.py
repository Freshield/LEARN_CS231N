import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print np.mean(x, axis=0)
print np.mean(x, axis=1)

a = np.ones((x.shape[0], 1))
print a
print np.hstack((x,a))

