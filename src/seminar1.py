# Seminar 1
# Numpy fundamentals

import numpy as np


def random_matrix(n: int) -> np.array:
    """
    Make (n x n x 3) matrix with a random uniform distribution [0, 255]
    array type must be uint8
    :param n: matrix size
    :return: random n x n x 3 matrix
    """
    return np.array((np.random.uniform(0, 256,  size=(n, n, 3))), dtype = np.uint8)

def broadcast_array(a: np.array, n: int) -> np.array:
    """
    Broadcast 1D array to 2D matrix by repeating it n times
    :param a: 1D numpy array
    :param n: number of rows in output matrix
    :return: 2D matrix
    """
    current_matrix = a
    for i in range(n-1):
        current_matrix = np.vstack([current_matrix, a])
    return current_matrix

def inplace_operation(a: np.array, b: np.array) -> None:
    """
    Compute ((a+b)*(-a/2)) in place (without copy)
    :param a: matrix A
    :param b: matrix B
    :return: None
    """

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
    return a


def self_inners(a: np.array) -> np.array:
    """
    Given 2D array A.shape = (m, n).
    Compute inners along axis n and return (m, m) matrix
    :param a:
    :return: 2D array of inners product shape=(m, m)
    """
    return a
