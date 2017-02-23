import numpy as np

x = np.zeros((4,4))
print x

y = np.zeros(4)
print y

x1 = np.linspace(0, 1, 11)
print x1

print 1e-0

a = np.array([[1,2,3],[4,5,6]])
b = np.array([1,2])
c = a + 0
c[range(2),b] -= 1
d = (a[range(2),b] - 1)

print a
print b
print c
print d

print np.random.choice(5, 7)
print np.random.choice(5, 5, replace=False)