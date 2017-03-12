import numpy as np

a = np.ones((1,2,3,4))
print a

b = np.pad(a,((0,0),(0,0),(2,2),(3,3)),'constant',constant_values=0)
print b

c = np.zeros((1,2,3,4))
print c

d = np.ones((1,2,3,4))
print d.shape
e,f = d.shape[-2:]
print e
print f

_, _, _, g = d.shape
print g

h = np.array([[1,2],[3,4]])
print h
h = np.pad(h,(1,1),'constant',constant_values=0)
print h
h = h[1:-1,1:-1]
print h

print np.maximum(1,2)
print np.max([1,2,3])