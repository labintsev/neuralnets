"""Seminar 3. Multilayer neural net"""
import numpy as np


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


def softmax_with_cross_entropy(Z, y):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      Z - predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      y - target_index, np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      d_out, np array same shape as predictions - gradient of predictions by loss value
    """
    N = Z.shape[0]
    Z -= np.max(Z)
    exp_scores = np.exp(Z)
    S = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    loss = - np.log(S[range(N), y]).mean()
    S[range(N), y] -= 1
    d_out = S / N
    return loss, d_out


def l2_regularization(W, reg_strength):
    loss = 0.5 * reg_strength * np.sum(W*W)
    grad = np.dot(W, reg_strength)
    return loss, grad


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X: np.array) -> np.array:
        """
        TODO: Implement forward pass
        Hint: you'll need to save some information about X
        in the instance variable to use it later in the backward pass
        :param X: input data
        :return: Rectified Linear Unit
        """
        raise Exception("Not implemented!")

    def backward(self, d_out: np.array) -> np.array:
        """
        Backward pass
        :param d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self) -> dict:
        # ReLU Doesn't have any parameters
        return {}


class DenseLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your implementation shouldn't have any loops
        raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        # raise Exception("Not implemented!")
        # print('d_out shape is ', d_out.shape)
        # print('self.W shape is ', self.W.value.shape)
        raise Exception("Not implemented!")

    def params(self):
        return {'W': self.W, 'B': self.B}


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg=0):
        """
        Initializes the neural network
        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.loss = None
        self.d_out = None
        self.layers = [DenseLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       DenseLayer(hidden_layer_size, n_output)
                       ]

    def forward(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        Z = X.copy()

        # TODO forward passes through the all model`s layer
        # Set layer parameters gradient to zeros
        # After that compute loss and gradients
        for layer in self.layers:
            for param in layer.params().values():
                pass

        self.loss, self.d_out = softmax_with_cross_entropy(Z, y)
        return Z

    def backward(self):
        # Update gradients of all layers,
        # implement l2 regularization on all params
        # Hint: self.params() is useful again!

        tmp_d_out = self.d_out
        for layer in reversed(self.layers):
            tmp_d_out = layer.backward(tmp_d_out)
            for param in layer.params().values():
                pass

    def fit(self, X, y, learning_rate=1e-3, num_iters=10000,
            batch_size=4, verbose=True):
        """
        Train classifier with stochastic gradient descent
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            idxs = np.random.choice(num_classes, batch_size)
            X_batch, y_batch = X[idxs], y[idxs]
            # evaluate loss and gradient
            self.forward(X_batch, y_batch)
            self.backward()

            # Update gradients
            for layer in self.layers:
                for param in layer.params().values():
                    param.value -= param.grad * learning_rate

            loss_history.append(self.loss)

            if it % 100 == 0 and verbose:
                print(f'iteration {it} / {num_iters}: loss {self.loss:.3f} ')

        return loss_history


if __name__ == '__main__':
    """1 point"""
    # Train your TwoLayer Net! 
    # Test accuracy must be > 0.33
    # Save report to output/seminar3
    model = TwoLayerNet()
