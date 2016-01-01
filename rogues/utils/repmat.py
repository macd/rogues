import numpy as np


class Higham(Exception):
    pass


def repmat(a, nrows, ncols):
    """
    repmat(a, nrows, ncols)
    Simple implementation of m*lab's repmat function.
    """
    c = np.kron(np.ones((nrows, ncols)), a)
    return c
