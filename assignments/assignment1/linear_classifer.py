import numpy as np
import math


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    scores = predictions - np.max(predictions, axis=1, keepdims=True)
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    probs = np.exp(scores)/sum_exp_scores
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    loss = 0.0
    num_batch = probs.shape[0]
    for i in range(num_batch):
      loss += -np.log(probs[i][target_index[i]])
    return loss
    


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    num_batch = predictions.shape[0]
    num_classes = predictions.shape[1]
    dprediction = np.zeros((num_batch, num_classes), np.float32)
    for i in range(num_batch):
      dprediction[i][target_index[i]] = 1
    soft = softmax(predictions)
    loss = cross_entropy_loss(soft, target_index)
    dprediction = soft - dprediction
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W * W)
    grad = reg_strength * 2 * W 

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    num_train = X.shape[0]
    predictions = np.dot(X, W)
    soft = softmax(predictions)
    loss = cross_entropy_loss(soft, target_index)
    soft[np.arange(num_train), target_index] -= 1
    dW = np.dot(X.T, soft)

    loss /= num_train
    dW /= num_train

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None


    def batch_generator(self, X, Y, batch_size = 100):
      indices = np.arange(len(X)) 
      batch=[]
      while True:
        # it might be a good idea to shuffle your data before each epoch
        np.random.shuffle(indices) 
        for i in indices:
          batch.append(i)
          if len(batch)==batch_size:
            return X[batch], Y[batch]
            batch=[]

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            X_batch = None
            y_batch = None
            '''
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            '''
            X_batch, y_batch = self.batch_generator(X, y, batch_size=batch_size)
            loss_reg, grad_reg = l2_regularization(self.W, reg)
            loss, grad = linear_softmax(X_batch, self.W, y_batch)
            loss += loss_reg
            grad += grad_reg
            self.W = self.W - learning_rate * (grad)
            loss_history.append(loss)
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history


    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        scores = X.dot(self.W)
        y_pred = scores.argmax(axis=1)

        return y_pred
