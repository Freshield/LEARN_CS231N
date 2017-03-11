import numpy as np

print np.random.rand(5)
print np.random.randn(5)
print np.random.rand(2,3)
print np.random.rand(2, 3) < 0.5

prop = 0.3
p = np.random.rand(20, 30)
mask = np.random.rand(20,30) < prop

print np.sum(p)
print np.sum(p * mask)
print np.sum(p * mask) / prop