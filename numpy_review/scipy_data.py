from scipy.misc import imread, imsave, imresize

img = imread('cat.jpg')
print img.dtype, img.shape

img_tinted = img * [1,0.95,0.9]
img_tinted = imresize(img_tinted, (300,300))

imsave('cat_tinted.jpg', img_tinted)

from scipy.spatial.distance import pdist, squareform
import numpy as np

x = np.array([[0,1],[1,0],[2,0]])
print x
d = squareform(pdist(x, 'euclidean'))
print d
print

