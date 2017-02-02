import numpy as np

a = np.array([1,2,3])
print type(a)
print a.shape
print 'ok'
print a[0],a[1],a[2]
a[0] = 5
print a

b = np.array([[1,2,3],[4,5,6]])
print b
print b.shape
print b[0,0], b[0,1], b[1,0]

a = np.zeros((2,2))
print a

b = np.zeros((1,2))
print b

c = np.full((2,2), 7)
print c

d = np.eye(2)
print d

e = np.random.random((2,2))
print e

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
print b
print a[0,1]
b[0,0] = 77
print a[0,1]
b[0,0] = 2

row_r1 = a[1, :]
row_r2 = a[1:2, :]
print row_r1, row_r1.shape
print row_r2, row_r2.shape
print row_r1[0]
print row_r2[0]

col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print col_r1, col_r1.shape
print col_r2, col_r2.shape

a = np.array([[1,2], [3,4], [5,6]])
print a[[0,1,2], [0,1,0]]
print np.array([a[0,0],a[1,1],a[2,0]])

print a[[0,0],[1,1]]
print np.array([a[0,1],a[0,1]])

a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
b = np.array([0,2,0,1])
print a[np.arange(4), b]
a[np.arange(4), b] += 10
print a

a = np.array([[1,2],[3,4],[5,6]])
bool_idx = (a > 2)
print bool_idx
print a[bool_idx]
print a[a>2]
