import numpy as np

a = np.arange(-2.0, 2.1, 0.1,dtype=np.float32)

print a

for b in a:
    print np.tanh(b)

for c in xrange(4, -1, -1):
    print c