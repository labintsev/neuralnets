import numpy as np


def softmax(Z: np.array) -> np.array:
    """
    TODO 1:
    Compute softmax of 2D array Z along axis -1
    :param Z: 2D array, shape (N, C)
    :return: softmax 2D array, shape (N, C)
    """
    e_Z = np.exp(Z)
    sum_Z = np.sum(e_Z, axis=1, keepdims=True)
    return e_Z / sum_Z

def main():
    x = np.ones((4, 3073)) * 100
    W = np.ones((3073, 2)) * 1e-3
    y = np.array([0, 0, 0, 0], dtype=int)
    yi = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    print(yi)
    print(yi[np.arange(len(x)), y])


if __name__ == '__main__':
    main()
