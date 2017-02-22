import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')

for j in xrange(K):
    ix = range(N*j, N*(j+1)) # 0-99 100-199 200-299
    r = np.linspace(0.0, 1, N) # 0-100 separte
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N) * 0.2 #0-4 4-8 8-12 separte
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

fig = plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()
fig.savefig('spiral_raw.png')
