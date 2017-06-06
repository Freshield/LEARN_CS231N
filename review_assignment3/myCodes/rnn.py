import numpy as np

a = np.arange(24).reshape([4,3,2])

print a

for t in range(3):
    print a[:,t,:]
    print a[:,t,:].shape
