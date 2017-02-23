import  matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8')

num_class = K
num_train = N * K

reg = 1e-3
learning_rate = 1e-0

for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

#show the raw image
fig = plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
fig.savefig('spiral_raw.png')

# X (300,2) y(300,1), W(2,3), b(1,3), socres(300,3), probs(300,3)

W = np.random.randn(D,num_class) * 0.01
b = np.zeros((1,num_class))

for i in xrange(200):
    # forward
    scores = np.dot(X, W) + b
    # f(x)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_class_socres = probs[range(num_train), y]
    data_loss = np.sum(-np.log(correct_class_socres)) / num_train
    # r(w)
    reg_loss = 0.5 * reg * np.sum(W * W)
    # loss
    loss = data_loss + reg_loss

    if i % 10 == 0:
        print "iteration %d: loss %f" % (i,loss)

    # backward
    pk = probs + 0
    pk[range(num_train), y] -= 1
    dscores = pk / num_train

    drw = 2 * 0.5 * reg * W

    db = np.sum(dscores, axis=0)

    dW = np.dot(X.T, dscores)

    dW += drw

    # update
    W -= learning_rate * dW
    b -= learning_rate * db

scores = np.dot(X, W) + b
predict_y = np.argmax(scores, axis=1)

print 'training accuracy: %.4f' % (np.mean(predict_y == y))

#show the softmax image
h = 0.02
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_linear.png')
plt.show()