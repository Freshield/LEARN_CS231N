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


# X (300,2) y(300,1), W(2,100), b(1,100), W2(100,3), b2(1,3), socres(300,3), probs(300,3)
# hidden(300,100)
num_hid = 100

W = np.random.randn(D, num_hid) * 0.01
b = np.zeros((1, num_hid))
W2 = np.random.randn(num_hid, num_class) * 0.01
b2 = np.zeros((1, num_class))

for i in xrange(10000):
    # forward


    hidden = np.maximum(0, np.dot(X, W) + b)  # (300,100)
    scores = np.dot(hidden, W2) + b2

    # calculate loss
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_class_scores = probs[range(num_train), y]
    data_loss = np.sum(-np.log(correct_class_scores)) / num_train
    reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss

    #print
    if i % 1000 == 0:
        print "iteration %d: loss %f" % (i,loss)

    # backward
    pk = probs
    pk[range(num_train), y] -= 1
    dscores = pk / num_train

    dW2 = np.dot(hidden.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden <= 0] = 0

    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)

    dW += 2 * 0.5 * reg * W
    dW2 += 2 * 0.5 * reg * W2

    # upgrade
    W -= learning_rate * dW
    b -= learning_rate * db
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

#test
hidden = np.maximum(0, np.dot(X, W) + b)  # (300,100)
scores = np.dot(hidden, W2) + b2
y_pred = np.argmax(scores, axis=1)
accuracy = np.mean(y_pred == y)
print 'training accuracy: %.4f' % accuracy

#show image
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_net.png')
plt.show()