import numpy as np


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


def softmax_with_cross_entropy(Z, y):
    N = Z.shape[0]
    Z -= np.max(Z)
    exp_scores = np.exp(Z)
    S = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    loss = -np.log(S[range(N), y]).mean()
    S[range(N), y] -= 1
    d_out = S / N
    return loss, d_out


class ReLULayer:
    def __init__(self):
        self.mask = None

    def forward(self, X: np.array) -> np.array:
        self.mask = X > 0
        return X * self.mask

    def backward(self, d_out: np.array) -> np.array:
        return d_out * self.mask

    def params(self) -> dict:
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
        d_input = np.dot(d_out, self.W.value.T)
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


class TwoLayerNet:
    def __init__(self, n_input, n_output, hidden_layer_size, reg=0):
        self.reg = reg
        self.loss = None
        self.d_out = None
        self.layers = [DenseLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       DenseLayer(hidden_layer_size, n_output)
                       ]

    def forward(self, X, y):
        Z = X.copy()

        for layer in self.layers:
            Z = layer.forward(Z)

        self.loss, self.d_out = softmax_with_cross_entropy(Z, y)
        return Z

    def backward(self):
        tmp_d_out = self.d_out
        for layer in reversed(self.layers):
            tmp_d_out = layer.backward(tmp_d_out)

    def fit(self, X, y, learning_rate=1e-3, num_iters=10000,
            batch_size=4, verbose=True):
        num_classes = np.max(y) + 1

        loss_history = []
        for it in range(num_iters):
            idxs = np.random.choice(len(X), batch_size)
            X_batch, y_batch = X[idxs], y[idxs]
            self.forward(X_batch, y_batch)
            self.backward()

            for layer in self.layers:
                for param in layer.params().values():
                    param.value -= learning_rate * (param.grad + self.reg * param.value)

            loss_history.append(self.loss)

            if it % 100 == 0 and verbose:
                print(f'iteration {it} / {num_iters}: loss {self.loss:.3f} ')

        return loss_history


if __name__ == '__main__':
    """1 point"""
    model = TwoLayerNet(3072, 10, 128, reg=0.0)

    loss_history = model.fit(X_train, y_train, learning_rate=1e-3, num_iters=10000, batch_size=4, verbose=True)


