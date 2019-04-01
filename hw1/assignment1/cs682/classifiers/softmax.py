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
  dW = dW.T
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores =  X.dot(W)
  num_classes = W.shape[1]
  normed_scores = np.zeros(scores.shape)
  for i in range(num_train):
      max_score = 0
      for j in range(num_classes):
          if scores[i][j] > max_score:
              max_score = scores[i][j] 
      norm_sum = 0
      for j in range(num_classes):
          normed_scores[i][j] = np.exp(scores[i][j]-max_score)
          norm_sum += normed_scores[i][j]
      for j in range(num_classes):
          normed_scores[i][j] /= norm_sum
          if j != y[i]:
            dW[j] +=  X[i].dot(normed_scores[i][j])
          else:
            dW[j] +=  X[i].dot(normed_scores[i][j]-1)
      loss += -np.log(normed_scores[i][y[i]])          
  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW = dW.T
  dW += 2*reg*W
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
  scores =  X.dot(W)
  
  max_scores = np.max(scores, axis = 1)
  max_scores =  max_scores.reshape(num_train,1)
  norm_scores = scores - max_scores
  norm_scores = np.exp(norm_scores)
  sum_exp_scores = np.sum(norm_scores, axis = 1)
  norm_scores = norm_scores/sum_exp_scores.reshape(num_train,1)
  loss = np.sum(-np.log(norm_scores[range(num_train),y]))
  loss /= num_train
  loss += reg * np.sum(W*W)

  norm_scores[range(num_train), y] -= 1
  dW += X.T.dot(norm_scores)
  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

