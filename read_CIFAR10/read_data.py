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
