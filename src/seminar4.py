"""Seminar 4. Batch normalization and Dropout. Advanced optimization."""

from abc import ABC, abstractmethod
from copy import deepcopy

from seminar3 import *
from test_utils import get_preprocessed_data

epsilon = 1e-3


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def params(self) -> dict:
        pass


class Optimizer(ABC):

    @abstractmethod
    def step(self, w, d_w, learning_rate):
        pass


class SGD(Optimizer):
    def step(self, w, d_w, learning_rate):
        # TODO Update W with d_W
        w -= d_w * learning_rate
        # pass


class Momentum(Optimizer):
    def __init__(self, rho=0.9):
        self.rho = rho
        self.velocity = None

    def step(self, w, d_w, learning_rate):
        if self.velocity is None:
            self.velocity = np.zeros_like(d_w)
        # TODO Update W with d_W and velocity
        new_velocity = self.rho * self.velocity + d_w
        w -= new_velocity * learning_rate
        self.velocity = new_velocity


class DropoutLayer(Layer):
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        if train:
            # TODO zero mask in random X position and scale remains
            # pass
            self.mask = np.random.random(size=x.shape) > self.p
            self.scale = 1 / (1 - self.p)
            return x * self.scale * self.mask
        else:
            return x

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        return grad_input * self.mask * self.scale

    def params(self) -> dict:
        return {}

    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.scale = None


class BatchNormLayer(Layer):
    def __init__(self, dims: int) -> None:
        self.gamma = Param(np.ones((1, dims), dtype="float32"))
        self.bias = Param(np.zeros((1, dims), dtype="float32"))

        self.running_mean_x = np.zeros(0)
        self.running_var_x = np.zeros(0)

        # forward params
        self.var_x = np.zeros(0)
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.num_examples = 0
        self.mean_x = np.zeros(0)
        self.running_avg_gamma = 0.9

    def _update_running_variables(self) -> None:
        is_mean_empty = np.array_equal(np.zeros(0), self.running_mean_x)
        is_var_empty = np.array_equal(np.zeros(0), self.running_var_x)
        if is_mean_empty != is_var_empty:
            raise ValueError("Mean and Var running averages should be "
                             "initialized at the same time")
        if is_mean_empty:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
        else:
            gamma = self.running_avg_gamma
            self.running_mean_x = gamma * self.running_mean_x + (1.0 - gamma) * self.mean_x
            self.running_var_x = gamma * self.running_var_x + (1. - gamma) * self.var_x

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.num_examples = x.shape[0]
        if train:
            # TODO Compute mean_x and var_x
            self.mean_x = np.mean(x, axis=0, keepdims=True)
            self.var_x = np.var(x, axis=0, keepdims=True)
            self._update_running_variables()
        else:
            # TODO Copy mean_x and var_x from running variables
            self.mean_x = self.running_mean_x
            self.var_x = self.running_var_x
            pass

        self.var_x += epsilon
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = x - self.mean_x
        self.standard_x = self.x_minus_mean / self.stddev_x
        return self.gamma.value * self.standard_x + self.bias.value

    def backward(self, grad_input: np.ndarray) -> np.ndarray:
        standard_grad = grad_input * self.gamma.value

        var_grad = np.sum(standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3 / 2),
                          axis=0, keepdims=True)
        stddev_inv = 1 / self.stddev_x
        aux_x_minus_mean = 2 * self.x_minus_mean / self.num_examples

        mean_grad = (np.sum(standard_grad * -stddev_inv, axis=0, keepdims=True) +
                     var_grad * np.sum(-aux_x_minus_mean, axis=0, keepdims=True))

        self.gamma.grad = np.sum(grad_input * self.standard_x, axis=0,
                                 keepdims=True)
        self.bias.grad = np.sum(grad_input, axis=0, keepdims=True)

        return standard_grad * stddev_inv + var_grad * aux_x_minus_mean + \
            mean_grad / self.num_examples

    def params(self) -> dict:
        return {'Gamma': self.gamma, 'bias': self.bias}


class NeuralNetwork:
    """ Neural network with two fully connected layers """

    def __init__(self, layers):
        self.loss = None
        self.d_out = None
        self.optimizers = None
        self.layers = layers

    def forward(self, X):
        Z = X.copy()
        for layer in self.layers:
            Z = layer.forward(Z)
            for param in layer.params().values():
                param.grad = np.zeros_like(param.grad)
        return Z

    def compute_loss_and_gradient(self, x, y):
        self.loss, self.d_out = softmax_with_cross_entropy(x, y)

    def setup_optimizer(self, optimizer):
        self.optimizers = dict()
        for param_name in self.params():
            self.optimizers[param_name] = deepcopy(optimizer)

    def backward(self, reg=0.0):
        tmp_d_out = self.d_out.copy()
        for layer in reversed(self.layers):
            tmp_d_out = layer.backward(tmp_d_out)
            if isinstance(layer, DenseLayer):
                for param in layer.params().values():
                    reg_loss, reg_grad = l2_regularization(param.value, reg)
                    param.grad += reg_grad
                    self.loss += reg_loss

    def fit(self, X, y, learning_rate=1e-3, num_iters=10000, batch_size=4, verbose=True):
        num_classes = np.max(y) + 1
        loss_history = []
        for it in range(num_iters):
            idxs = np.random.choice(num_classes, batch_size)
            X_batch, y_batch = X[idxs], y[idxs]
            # evaluate loss and gradient
            z = self.forward(X_batch)
            self.compute_loss_and_gradient(z, y_batch)
            self.backward()

            # Update gradients

            for name, param in self.params().items():
                self.optimizers[name].step(param.value, param.grad, learning_rate)

            loss_history.append(self.loss)

            if it % 100 == 0 and verbose:
                print(f'iteration {it} / {num_iters}: loss {self.loss:.3f} ')

        return loss_history

    def params(self):
        model_params = dict()
        for i, layer in enumerate(self.layers):
            layer_params = layer.params()
            for k, v in layer_params.items():
                model_params[f'layer_{i}_{k}'] = v
        return model_params

    def evaluate(self, X, y):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points and evaluate accuracy.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        Returns:
        - y_predicted: Predicted labels for the data in X. y_predicted is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        z = self.forward(X)
        y_predicted = np.argmax(z, axis=1)
        accuracy = np.mean(y_predicted == y)
        return accuracy

def train_():
    (x_train, y_train), (x_test, y_test) = get_preprocessed_data(include_bias=False)
    hidden_layer_size = 256
    n_input = 3072
    n_output = 10

    reg = 0.5
    learning_rate = 1e-4
    num_iters = 5_000
    batch_size = 16

    neural_net = NeuralNetwork([DenseLayer(n_input, hidden_layer_size),
                                DropoutLayer(0.5),
                                BatchNormLayer(hidden_layer_size),
                                ReLULayer(),
                                DenseLayer(hidden_layer_size, n_output)])
    neural_net.setup_optimizer(SGD())
    t0 = datetime.datetime.now()

    loss_history = neural_net.fit(x_train, y_train, learning_rate, num_iters, batch_size)
    t1 = datetime.datetime.now()
    dt = t1 - t0

    report = f"""# Training Softmax classifier  
    datetime: {t1.isoformat(' ', 'seconds')}  
    Well done in: {dt.seconds} seconds  
    learning_rate = {learning_rate}  
    reg = {reg}  
    num_iters = {num_iters}  
    batch_size = {batch_size}  

    Final loss: {loss_history[-1]:.6f}   
    Train accuracy: {neural_net.evaluate(x_train, y_train)}   
    Test accuracy: {neural_net.evaluate(x_test, y_test)}  

    <img src="weights.png">  
    <br>
    <img src="loss.png">
    """

    print(report)
    with open('seminar4/report.md', 'w') as f:
        f.write(report)
    visualize_loss(loss_history, 'seminar4')

if __name__ == '__main__':
    """1 point"""
    # Train your neural net!
    train_()


