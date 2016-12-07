def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

data = unpickle('data_batch_1')

print data['data'].shape
print len(data['labels'])

import numpy as np

a = np.full((32,32),1)
print a.shape

b = np.array([1,2,3])
print b.shape

a = np.full((32,32,3),b)
print a.shape
print a[0,0,2]

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8,9],[10,11,12]])
print a[0] - b[0]
print a[0,:] - b[0,:]
print np.sqrt(np.sum(np.square(a[0,:] - b[0,:])))

print a.shape, b.shape
print b - a[0]
print b - a[0,:]
print np.sum(b - a[0], axis=1)

x = np.arange(60,step=3)
print x
y = np.array([1,3,5,7,9])
print x[y[:]]

t = np.array([1,23,1,4,5,1])
print np.bincount(t)
print 1 in t

dic = {}
for item in t:
    if dic.get(item,-1) == -1:
        dic[item] = 1
    else:
        dic[item] += 1

print dic

big = -1
num = -1
for item in dic:
    now = dic[item]
    if now > big:
        num = item
        big = now

print num