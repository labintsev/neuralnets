# Seminar 1
# Numpy fundamentals

import numpy as np


def random_matrix(n: int) -> np.array:
    random_values = np.random.uniform(low=0, high=255, size=(n, n, 3))
    random_m = random_values.astype(np.uint8)
    return random_m
    """
    Make (n x n x 3) matrix with a random uniform distribution [0, 255]
    array type must be uint8
    :param n: matrix size
    :return: random n x n x 3 matrix
    """
     # return np.random.randint(0, 255, size=(n, n, 3), dtype=np.uint8)


def broadcast_array(a: np.array, n: int) -> np.array:
    matrix_2D = np.repeat(a[None, :], n, axis=0)
    return matrix_2D
    """
    Broadcast 1D array to 2D matrix by repeating it n times
    :param a: 1D numpy array
    :param n: number of rows in output matrix
    :return: 2D matrix
    """
    # return np.ones(n).reshape(-1, 1) * a.reshape(1, -1)


def inplace_operation(a: np.array, b: np.array) -> None:

    """
    Compute ((a+b)*(-a/2)) in place (without copy)
    :param a: matrix A
    :param b: matrix B
    :return: None
    """
    t = ((a + b) * (-a / 2))
    a[...] = t[...]

    # a += b
    # a *= -1
    # a /= 2
def get_elements(a: np.array, indices: np.array) -> np.array:
    """
    Given 2D matrix of elements and 1D array of indexes.
    Return elements of each row in matrix with index i.
    For example:
     A = [
     [0,1,2],
     [3,4,5],
     [6,7,8]
     ]
     i = [0, 1, 2]
    Expected: get_elements(A, i) = [0, 4, 8]
    :param a: 2D array
    :param indices: 1D array
    :return: 1D array of elements
    """
    return a[np.arange(0, a.shape[0]), indices]
    # N = len(a)
    # return a[range(N), indices]
def self_inners(a: np.array) -> np.array:
    """
    Given 2D array A.shape = (m, n).
    Compute inners along axis n and return (m, m) matrix
    :param a:
    :return: 2D array of inners product shape=(m, m)
    """
    return a @ a.T

