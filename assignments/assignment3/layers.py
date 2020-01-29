import numpy as np


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
    predictions -= np.max(predictions)
    res = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    return res


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
    res = - np.log(probs[np.arange(len(probs)), target_index])
    return res


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = (W * W).sum() * reg_strength
    grad = 2 * W * reg_strength
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
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
    """
    predictions = predictions.copy()

    probs = softmax(predictions)

    loss = cross_entropy_loss(probs, target_index).mean()

    mask = np.zeros_like(predictions)
    mask[np.arange(len(mask)), target_index] = 1

    dprediction = - (mask - softmax(predictions)) / mask.shape[0]

    return loss, dprediction



class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        result = np.maximum(X, 0)
        self.X = X
        return result

    def backward(self, d_out):
        d_result = (self.X > 0) * d_out
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = Param(X)
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        X = self.X.value
        W = self.W.value

        d_W = np.dot(X.T, d_out)
        d_B = np.dot(np.ones((X.shape[0], 1)).T, d_out)
        d_input = np.dot(d_out, W.T)

        self.W.grad += d_W
        self.B.grad += d_B

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.stride = None
        self.X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.stride = 1
        s = self.stride
        out_height = int(1 + (height + 2 * self.padding - self.filter_size) / self.stride)
        out_width = int(1 + (width + 2 * self.padding - self.filter_size) / self.stride)

        pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)

        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for oh in range(out_height):
            for ow in range(out_width):
                for bs in range(batch_size):
                    for oc in range(self.out_channels):
                        out[bs, oh, ow, oc] = np.sum(X[bs, oh * s:oh * s + self.filter_size,
                                                     ow * s:ow * s + self.filter_size, :] *
                                                     self.W.value[:, :, :, oc]) + self.B.value[oc]

        self.X = X
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dW = np.zeros(shape=(batch_size, out_height, out_width, self.out_channels))
        dB = np.ones_like(self.out_channels, dtype=float)
        dX = np.zeros_like(self.X)

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        dB = np.sum(d_out, (0, 1, 2))
        for y in range(out_height):
            for x in range(out_width):
                dW += np.sum(self.X[:, y, x, :].T * d_out[:, y, x, :])
                dX += np.sum(d_out[:, y, x, :]*self.W.value[:, y, x, :].T)
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                pass
        self.W.grad += dW
        self.B.grad += dB
        return dX

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
