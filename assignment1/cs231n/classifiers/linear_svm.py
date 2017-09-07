import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # L = data loss  + regularization loss
  # L = (1/N) * 𝝨 (𝝨 (j ≠ yi)(max(0, f[j] - f[yi] + 1))) + λR(W)

  # data loss = (1/N) * 𝝨 (j ≠ yi)(max(0, f[j] - f[yi] + 1)
  # regularization loss = λR(W)

  # f(x) = x * W
  # R(W) = 𝝨k 𝝨l (W^2<k,l>)

  # compute the loss and the gradient
  # http://cs231n.github.io/optimization-1/#gradcompute

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # f(j) - f(yi) + 1
      if margin > 0:                               # max(0, margin)
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] += -X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  loss /= num_train
  # Average gradients as well
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  # Add regularization to the gradient
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W) # 500 x 10
  correct_scores = scores[np.arange(X.shape[0]), y][:, None] # 500 * 1
  deltas = 1

  margins = scores - correct_scores + deltas
  margins = np.maximum(0, margins)
  margins[np.arange(X.shape[0]), y] = 0 # j == y[i]

  loss = np.sum(margins)

  # Average
  num_train = X.shape[0]
  loss /= num_train

  # Regularization
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # if margin > 0 add class j by X and subtract class y[i] by X
  slopes = np.zeros_like(margins)
  slopes[margins > 0] = 1
  slopes[range(num_train), y] -= np.sum(margins > 0, axis = 1)
  dW = X.T.dot(slopes)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
