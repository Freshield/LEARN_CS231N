import numpy as np

a = np.array([[0,1,2,3,4,5],[1,1,1,1,1,1]])

print np.prod(a,axis=0)

b = np.linspace(-1,5,7)

print b

c = np.array([1,2,3])
d = c
e = np.copy(c)
print c
print d
print e
d[0] = 10
print c
print d
print e