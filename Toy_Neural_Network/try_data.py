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

x = [2,3,1,5,7,1,4]
print np.argmin(x)
print x.pop(np.argmin(x))
print x

print range(10)
print range(1,10,1)

x = np.reshape(x, (1,-1))
y = 4
mask = x > 4
print mask
print x[x > y]
"""
KeyPress event, serial 37, synthetic NO, window 0x6000001,
    root 0xf6, subw 0x0, time 9198554, (501,91), root:(1670,163),
    state 0x0, keycode 92 (keysym 0xfe03, ISO_Level3_Shift), same_screen YES,
    XLookupString gives 0 bytes:
    XmbLookupString gives 0 bytes:
    XFilterEvent returns: False

KeyRelease event, serial 37, synthetic NO, window 0x6000001,
    root 0xf6, subw 0x0, time 9198727, (501,91), root:(1670,163),
    state 0x80, keycode 92 (keysym 0xfe03, ISO_Level3_Shift), same_screen YES,
    XLookupString gives 0 bytes:
    XFilterEvent returns: False

"""