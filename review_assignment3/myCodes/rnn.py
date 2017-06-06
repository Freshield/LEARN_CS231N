import numpy as np

a = np.arange(24).reshape([4,3,2])

print a

for t in range(3):
    print a[:,t,:]
    print a[:,t,:].shape

for t in range(2, -1, -1):
    print t

b = np.array([1,0,1,1,0,0]).reshape((2,3))

w = np.array([1,2,3,4,5,6,7,8]).reshape((2,4))

print w[b,:]
