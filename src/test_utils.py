# Seminar 2. Testing utils.
import os.path
import numpy as np
import matplotlib.pyplot as plt


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    """
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    """

    assert isinstance(x, np.ndarray)

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        delta_vec = np.zeros(x.shape)
        delta_vec[ix] = delta
        numeric_grad_at_ix = (f(x + delta_vec)[0] - f(x - delta_vec)[0]) / (2 * delta)

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (
            ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False
        it.iternext()

    print("Gradient check passed!")
    return True


def get_preprocessed_data(include_bias=True):
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    x_train = x_train.astype(np.float16).reshape(x_train.shape[0], -1)
    x_test = x_test.astype(np.float16).reshape(x_test.shape[0], -1)

    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_train /= 128.0
    x_test -= mean_image
    x_test /= 128.0

    if include_bias:
        # Stack X with dummy feature 1 allows does not care about adding bias.
        # Now bias b is a part of a matrix W (D+1, C).
        # x @ W + b  => x' @ W'
        x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
        x_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])

    return (x_train, y_train), (x_test, y_test)


def visualize_weights(cls, out_dir='output/seminar2'):
    w = cls.W[:-1, :]  # strip out the bias
    w = w.reshape(32, 32, 3, 10)

    w_min, w_max = np.min(w), np.max(w)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        w_img = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(w_img.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.savefig(os.path.join(out_dir, 'weights.png'))


def visualize_loss(loss_history, out_dir='output/seminar2'):
    fig1, ax1 = plt.subplots()
    ax1.plot(loss_history)
    ax1.set_title('Loss history')
    fig1.savefig(os.path.join(out_dir, 'loss.png'))


def check_layer_gradient(layer, x, delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the input and output of a layer

    Arguments:
      layer: neural network layer, with forward and backward functions
      x: starting point for layer input
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(x):
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        grad = layer.backward(d_out)
        return loss, grad

    return check_gradient(helper_func, x, delta, tol)


def check_model_gradient(model, X, y,
                         delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for all model parameters

    Arguments:
      model: neural network model with compute_loss_and_gradients
      X: batch of input data
      y: batch of labels
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Returns:
      bool indicating whether gradients match or not
    """
    for layer in model.layers:
        params = layer.params()
        for param_key in params:
            print("Checking gradient for %s" % param_key)
            param = params[param_key]
            initial_w = param.value

            def helper_func(w):
                param.value = w
                model.forward(X, y)
                model.backward()
                loss = model.loss
                grad = param.grad
                return loss, grad

            if not check_gradient(helper_func, initial_w, delta, tol):
                return False

    return True
