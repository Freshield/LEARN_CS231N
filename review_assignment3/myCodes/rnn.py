import numpy as np

a = np.arange(24).reshape([4,6])

print a

print np.sum(a, axis=0)
