"""Seminar 3. Multilayer neural net"""
import datetime
import os

import numpy as np

from src.test_utils import visualize_weights, visualize_loss, get_preprocessed_data


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
    N = Z.shape[0]   # number of examples
    Z -= np.max(Z)  # to stabilize the exponent calculation
    exp_scores = np.exp(Z)
    S = exp_scores / exp_scores.sum(axis=1, keepdims=True)  # calculating values for the probability matrix S
    loss = - np.log(S[range(N), y]).mean()  # cross entropy for each class and its mean
    S[range(N), y] -= 1  # derivative of the probability matrix S
    d_out = S / N    # derivative of the output layer d_out
    return loss, d_out


def l2_regularization(W, reg_strength):
    loss = 0.5 * reg_strength * np.sum(W*W)
    grad = np.dot(W, reg_strength)
    return loss, grad

class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X: np.array) -> np.array:
        self.mask = (X > 0) # else 1, if x > 0
        return X * self.mask

    def backward(self, d_out: np.array) -> np.array:
        return d_out * self.mask

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
        # raise Exception("Not implemented!")
        self.X = X.copy()
        return X @ self.W.value + self.B.value

    def backward(self, d_out):
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        return d_out @ self.W.value.T

    def params(self):
        return {'W': self.W, 'B': self.B}


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output,
                 hidden_layer_size, reg=0):
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
                       DenseLayer(hidden_layer_size, n_output)]

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
            Z = layer.forward(Z)
            for param in layer.params().values():
                param.grad = np.zeros_like(param.grad)

        # compute loss and gradient:
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
                reg_loss, reg_grad = l2_regularization(param.value, self.reg)
                self.loss += reg_loss
                param.grad += reg_grad
    def evaluate(self, X, y):
        Z = X.copy()
        for layer in self.layers:
            Z = layer.forward(Z)
        predicted_classes = np.argmax(Z, axis=1)
        accuracy = np.mean(predicted_classes == y)
        return accuracy
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
    n_input = 3073
    n_output = 10
    n_hidden = 256

    # TODO 5: Find the best hyperparameters
    # assert test accuracy > 0.22
    # weights images must look like in lecture slides

    # ***** START OF YOUR CODE *****
    learning_rate = 1e-3
    reg = 1e-2
    num_iters = 1000
    batch_size = 64
    # ******* END OF YOUR CODE ************

    (x_train, y_train), (x_test, y_test) = get_preprocessed_data()
    cls = TwoLayerNet(n_input, n_output, n_hidden, reg)  # Убедитесь, что передаёте правильные параметры здесь
    t0 = datetime.datetime.now()
    loss_history = cls.fit(x_train, y_train, learning_rate, num_iters, batch_size, verbose=True)
    t1 = datetime.datetime.now()
    dt = t1 - t0

    report = f"""# Training Softmax classifier  
    datetime: {t1.isoformat(' ', 'seconds')}  
    Well done in: {dt.seconds} seconds  
    learning_rate = {learning_rate}  
    reg = {reg}  
    num_iters = {num_iters}  
    batch_size = {batch_size}  

    Final loss: {loss_history[-1]}   
    Train accuracy: {cls.evaluate(x_train, y_train)}   
    Test accuracy: {cls.evaluate(x_test, y_test)}  

    <img src="weights.png">  
    <br>
    <img src="loss.png">
    """

    print(report)

    out_dir = 'C://Python//neuralnets//output//seminar3'
    report_path = os.path.join(out_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(report)

    visualize_weights(cls, out_dir)
    visualize_loss(loss_history, out_dir)

