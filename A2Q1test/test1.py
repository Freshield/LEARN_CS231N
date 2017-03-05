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

print np.sum(b > 0)

f = [{'model':'train'} for i in xrange(5)]

print f

f[0]['model'] = 'test'
print f

for i in xrange(1,5):
    print i

for i in xrange(5):
    print i

for i in xrange(10, 0, -1):
    print i

g = {'name':'yy', 'sex':'male'}
g.setdefault('name','ww')
g.setdefault('age','26')

print g

h = {'name':'yy', 'sex':'male'}

print h.get('name','18')
print h.get('dream','win')
print h
