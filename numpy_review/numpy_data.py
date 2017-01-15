import numpy as np

a = np.array([1,2,3])
print type(a)
print a.shape
print a[0], a[1], a[2]
a[0] = 5
print a
print

b = np.array([[1,2,3],[4,5,6]])
print b
print b.shape
print b[0,0],b[0,1],b[1,0]
print

a = np.zeros((2,2))
print a
a = np.zeros(b.shape)
print a
print

b = np.ones((1,2))
print b
print

c = np.full((2,2), 7)
print c
print

d = np.eye(2)
print d
print

e = np.random.random((2,2))
print e
print

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print a
b = a[:2, 1:3]
print b
print a[0,1]
b[0,0] = 77
print a[0,1]

row_r1 = a[1, :]
row_r2 = a[1:2, :]
print row_r1, row_r1.shape
print row_r2, row_r2.shape
print
row_r1[0] = 99
print a
print row_r1
print row_r2
row_r1[0] = 5
a[0,1] = 2
print a.shape
print

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print col_r1, col_r1.shape
print col_r2, col_r2.shape
print

a1 = np.array([[1,2], [3,4], [5,6]])
print a1[[0,1,2],[0,1,0]]
print np.array([a1[0,0], a1[1,1], a1[2,0]])
print

print a[[0,0],[1,1]]
print np.array([a[0,1], a[0,1]])
print

print a
a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print a
b = np.array([0,2,0,1])
print a[np.arange(4), b]
a[np.arange(4), b] += 10
print a
print

a = np.array([[1,2],[3,4],[5,6]])
bool_idx = (a > 2)
print bool_idx
print a[bool_idx]
print a[a > 2]
print

x = np.array([1,2])
print x.dtype

x = np.array([1.0,2.0])
print x.dtype

x = np.array([1,2], dtype=np.int64)
print x.dtype
print

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

print x + y
print np.add(x,y)

print x - y
print np.subtract(x,y)

print x * y
print np.multiply(x,y)

print x / y
print np.divide(x,y)

print np.sqrt(x)
print

v = np.array([9,10])
w = np.array([11,12])

print v.dot(w)
print np.dot(v,w)
print np.dot(w,v)
print

print x.dot(v)
print np.dot(x, v)
print np.dot(v, x)

print x.dot(y)
print np.dot(x,y)
print

x = np.array([[1,2],[3,4]])

print np.sum(x)
print np.sum(x, axis=0)
print np.sum(x, axis=1)
print

print x
print x.T
print

v = np.array([1,2,3])
print v
print v.T
v = np.array([[1,2,3]])
print v
print v.T
print

