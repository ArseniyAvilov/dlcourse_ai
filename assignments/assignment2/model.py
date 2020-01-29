import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network
        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layer_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.non_linier = ReLULayer()
        self.layer_2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()
        W1 = params["W1"]
        B1 = params["B1"]
        W2 = params["W2"]
        B2 = params["B2"]

        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)


        f1 = self.layer_1.forward(X)
        f2 = self.non_linier.forward(f1)
        f3 = self.layer_2.forward(f2)
        loss, grad = softmax_with_cross_entropy(f3, y)

        d2 = self.layer_2.backward(grad)
        d_nl = self.non_linier.backward(d2)
        d1 = self.layer_1.backward(d_nl)

        l2_W1_loss, l2_W1_grad = l2_regularization(W1.value, self.reg)
        l2_B1_loss, l2_B1_grad = l2_regularization(B1.value, self.reg)
        l2_W2_loss, l2_W2_grad = l2_regularization(W2.value, self.reg)
        l2_B2_loss, l2_B2_grad = l2_regularization(B2.value, self.reg)

        l2_reg = l2_W1_loss + l2_B1_loss + l2_W2_loss + l2_B2_loss
        loss += l2_reg

        W1.grad += l2_W1_grad
        B1.grad += l2_B1_grad
        W2.grad += l2_W2_grad
        B2.grad += l2_B2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        y_pred = np.zeros(X.shape[0], np.int)

        out1 = self.layer_1.forward(X)
        out_relu = self.non_linier.forward(out1)
        predictions = self.layer_2.forward(out_relu)
        probs = softmax(predictions)
        y_pred = np.argmax(probs, axis=1)

        return y_pred

    def params(self):
        result = {'W1': self.layer_1.W, 'B1': self.layer_1.B,
                  'W2': self.layer_2.W, 'B2': self.layer_2.B}
        return result
