import numpy as np

a = np.arange(-2.0, 2.1, 0.1,dtype=np.float32)

print a

for b in a:
    print np.tanh(b)

for c in xrange(4, -1, -1):
    print c

d = np.zeros(5)
print d

d[0] = 1
print d

N, T, V, D = 2, 4, 5, 2

x = np.asarray([[1, 2], [0, 1]])
W = np.array([
    [0,1,2],
    [3,4,5],
    [6,7,8]
])
out = W[x,:]
print 'x'
print x
print 'W'
print W
print 'out'
print out
print 'dW'
dW = np.zeros_like(W)
np.add.at(dW, x, W[x,:])
print dW

dout = [[[0.1,0.0,0.2],[0.3,0.1,0.0]],[[0.2,0.1,0.1],[0.1,0.2,0.0]]]
dW2 = np.zeros_like(W, dtype=np.float64)
for index,idx in enumerate(x):

    dW2[idx] += dout[index]

print 'dW2',dW2

print 'a'


a = np.array([[1,2,3,4],[5,6,7,8]])
b = np.array([5,6])
np.add.at(a,[[0,1],[1,2]],b)
print a

print
