import numpy as np

x = np.array([[2,0,1],[3,4,0]])

W = np.arange(50).reshape((5,10))

print x
print W


print W[x,:]

print "here"

x = np.array([[1,0,3],[0,1,2]])

dout = np.array([[[1,1,1,1],[2,2,2,2],[3,3,3,3]],[[4,4,4,4],[6,6,6,6],[7,7,7,7]]])

dw = np.zeros((4,4))

np.add.at(dw, x, dout)

print dw