# Seminar 2. Softmax classifier.
import datetime
import os.path

import numpy as np
from src.test_utils import get_preprocessed_data, visualize_weights, visualize_loss


def softmax(X: np.array) -> np.array:
    exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
    softmax_X = exp_X / np.sum(exp_X, axis=-1, keepdims=True)
    return softmax_X
    """
    TODO 1:
    Compute softmax of 2D array along axis -1
    :param X: 2D array, shape (N, C)
    :return: softmax 2D array, shape (N, C)
    """


def softmax_loss_and_grad(W: np.array, X: np.array, y: np.array, reg: float) -> tuple:
    """
    TODO 2:
    Compute softmax classifier loss and gradient dL/dW
    Do not forget about regularization!
    :param W: classifier weights (D, C)
    :param X: input features (N, D)
    :param y: class labels (N, )
    :param reg: regularisation strength
    :return: loss, dW
    """
    loss = 0.0
    dL_dW = np.zeros_like(W)
    num_samples = X.shape[0]
    # *****START OF YOUR CODE*****
    # 1. Forward pass, compute loss as sum of data loss and regularization loss [sum(W ** 2)]
    scores = np.dot(X, W)  # Shape: (N, C)
    scores -= np.max(scores, axis=1, keepdims=True)  # For numerical stability
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Shape: (N, C)
    correct_logprobs = -np.log(probs[np.arange(num_samples), y])
    data_loss = np.sum(correct_logprobs) / num_samples
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss

    # 2. Backward pass, compute intermediate dL/dZ
    dscores = probs.copy()
    dscores[np.arange(num_samples), y] -= 1
    dscores /= num_samples

    # 3. Compute data gradient dL/dW
    dL_dW = np.dot(X.T, dscores)

    # 4. Compute regularization gradient
    dL_dW += reg * W

    # 5. Return loss and sum of data + reg gradients

    # *****END OF YOUR CODE*****

    return loss, dL_dW


class SoftmaxClassifier:
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-1, num_iters=10000,
              batch_size=64, verbose=True):
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
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is the number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch, y_batch = None, None

            #########################################################################
            # TODO 3:                                                               #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate batch_indices. Sampling with   #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = softmax_loss_and_grad(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO 4:                                                               #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            self.W -= learning_rate * grad

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            if it % 100 == 0:
                if verbose:
                    print(f'iteration {it} / {num_iters}: loss {loss:.3f} ')

        return loss_history

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
        z = X @ self.W
        y_predicted = np.argmax(z, axis=1)
        accuracy = np.mean(y_predicted == y)
        return accuracy


def train():
    # TODO 5: Find the best hyperparameters
    # assert test accuracy > 0.22
    # weights images must look like in lecture slides

    # ***** START OF YOUR CODE *****
    learning_rate = 1e-3  # You can start with a small learning rate
    reg = 1e-4  # You can start with a small regularization strength
    num_iters = 1000  # You can start with a small number of iterations
    batch_size = 64  # You can start with a small batch size
    # ******* END OF YOUR CODE ************

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
