import numpy as np

print np.ones((3,1))

a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2])
print a[:,b]
print a[np.arange(2),b]
