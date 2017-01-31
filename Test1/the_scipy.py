from scipy.misc import imread, imsave, imresize
import numpy as np

v = np.array([[1,4],[2,5],[3,6]])
w = np.array([4,5])
print v * w

img = imread('cat.jpg')

print img.dtype, img.shape

img_tinted = img * [1,0.95,0.9]

print img_tinted.shape

img_tinted = imresize(img_tinted,(300,300))

imsave('cat_tinted.jpg', img_tinted)