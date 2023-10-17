import datetime
import os.path

import numpy as np
from src.test_utils import get_preprocessed_data, visualize_weights, visualize_loss


def softmax(Z: np.array) -> np.array:
    """
    Compute softmax of 2D array Z along axis -1
    :param Z: 2D array, shape (N, C)
    :return: softmax 2D array, shape (N, C)
    """
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    softmax_output = e_Z / np.sum(e_Z, axis=1, keepdims=True)
    return softmax_output


def softmax_loss_and_grad(W: np.array, X: np.array, y: np.array, reg: float) -> tuple:
    loss = 0.0
    dL_dW = np.zeros_like(W)

    # 1. Forward pass, compute loss as sum of data loss and regularization loss [sum(W ** 2)]
    num_samples = X.shape[0]
    scores = X.dot(W)
    softmax_output = softmax(scores)

    # Calculate the data loss
    data_loss = -np.log(softmax_output[range(num_samples), y])
    loss = np.sum(data_loss) / num_samples

    # Add regularization loss
    reg_loss = 0.5 * reg * np.sum(W ** 2)
    loss += reg_loss

    # 2. Backward pass, compute intermediate dL/dZ
    dL_dZ = softmax_output.copy()
    dL_dZ[range(num_samples), y] -= 1
    dL_dZ /= num_samples

    # 3. Compute data gradient dL/dW
    dL_dW = X.T.dot(dL_dZ)

    # 4. Compute regularization gradient
    dL_dW += reg * W

    # 5. Return loss and the gradient
    return loss, dL_dW


class SoftmaxClassifier:
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-1, num_iters=10000,
              batch_size=64, verbose=True):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is the number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch, y_batch = None, None

            # Sample batch_size elements from the training data and their corresponding labels
            # Hint: Use np.random.choice to generate batch_indices. Sampling with replacement is faster.
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # evaluate loss and gradient
            loss, grad = softmax_loss_and_grad(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)

            # Update the weights using the gradient and the learning rate
            self.W -= learning_rate * grad

            if it % 100 == 0:
                if verbose:
                    print(f'iteration {it} / {num_iters}: loss {loss:.3f} ')

        return loss_history

    def evaluate(self, X, y):
        z = X @ self.W
        y_predicted = np.argmax(z, axis=1)
        accuracy = np.mean(y_predicted == y)
        return accuracy


def train():
    # Find the best hyperparameters
    learning_rate = 1e-3
    reg = 1e-1
    num_iters = 10000
    batch_size = 64

    (x_train, y_train), (x_test, y_test) = get_preprocessed_data()
    cls = SoftmaxClassifier()
    t0 = datetime.datetime.now()
    loss_history = cls.train(x_train, y_train, learning_rate, reg, num_iters, batch_size, verbose=True)
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

    out_dir = 'output/seminar2'
    report_path = os.path.join(out_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    visualize_weights(cls, out_dir)
    visualize_loss(loss_history, out_dir)


if __name__ == '__main__':
    train()



