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

x = np.asarray([[0, 3, 1, 2], [0, 1, 2, 3]])
W = np.array([
    ['I','','',''],
    ['L','O','V','E'],
    ['Y','O','U',''],
    ['!','','','']
])
print 'x'
print x
print 'W'
print W
print 'W[x,:]'
print W[x,:]