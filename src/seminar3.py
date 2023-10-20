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
    loss = 0.5 * reg_strength * np.sum(W * W)
    grad = reg_strength * W
    return loss, grad

class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X: np.array) -> np.array:
        self.mask = X > 0
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
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        dX = np.dot(d_out, self.W.value.T)
        dW = np.dot(self.X.T, d_out)
        dB = np.sum(d_out, axis=0, keepdims=True)
        self.W.grad += dW
        self.B.grad += dB
        return dX

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
        for layer in self.layers:
            for param in layer.params().values():
                param.grad = 0

        for layer in self.layers:
            Z = layer.forward(Z)

        self.loss, self.d_out = softmax_with_cross_entropy(Z, y)
        return Z

    def backward(self):
        tmp_d_out = self.d_out
        for layer in reversed(self.layers):
            tmp_d_out = layer.backward(tmp_d_out)
            for param in layer.params().values():
                l2_loss, l2_grad = l2_regularization(param.value, self.reg)
                param.grad += l2_grad

    def fit(self, X, y, learning_rate=1e-3, num_iters=10000,
            batch_size=4, verbose=True):
        num_classes = np.max(y) + 1
        loss_history = []
        for it in range(num_iters):
            idxs = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idxs], y[idxs]
            self.forward(X_batch, y_batch)
            self.backward()
            for layer in self.layers:
                for param in layer.params().values():
                    param.value -= param.grad * learning_rate

            loss_history.append(self.loss)

            if it % 100 == 0 and verbose:
                print(f'iteration {it} / {num_iters}: loss {self.loss:.3f} ')

        return loss_history

if __name__ == '__main__':
    model = TwoLayerNet(n_input=2, n_output=3, hidden_layer_size=10, reg=1e-4)

    # Generate some random data for testing
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.random.randint(0, 3, 100)

    # Train the model
    loss_history = model.fit(X, y, learning_rate=1e-3, num_iters=1000, batch_size=32, verbose=True)
