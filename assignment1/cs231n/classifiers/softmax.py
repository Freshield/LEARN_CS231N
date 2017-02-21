import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in xrange(num_train):
    scores = np.dot(X[i], W)
    exp_score = np.zeros_like(scores)
    P_scores = np.zeros_like(scores)
    #numerical stable
    scores -= np.max(scores)
    correct_class = y[i]
    row_sum = 0
    for j in xrange(num_class):
      exp_score[j] = np.exp(scores[j])
      row_sum += exp_score[j]
    for j in xrange(num_class):
      P_scores[j] = exp_score[j] / row_sum
      if j != correct_class:
        dW[:,j] += P_scores[j] * X[i]
      else:
        dW[:,j] += (P_scores[j] - 1) * X[i]
    loss += -np.log(P_scores[correct_class])

  loss = loss / num_train + 0.5 * reg * np.sum(W * W)
  dW = dW / num_train + 2 * 0.5 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = np.dot(X, W)
  #numerical stable
  scores -= np.max(scores, axis=1).reshape(-1,1)
  exp_scores = np.exp(scores)
  P_scores = exp_scores / np.sum(exp_scores, axis=1).reshape(-1,1)

  index = np.arange(num_train)
  correct_class = P_scores[index,y]
  loss = -np.sum(np.log(correct_class))
  loss = loss / num_train + 0.5 * reg * np.sum(W * W)

  P_scores[index,y] -= 1
  dW = np.dot(X.T, P_scores)
  dW = dW / num_train + 2 * 0.5 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

