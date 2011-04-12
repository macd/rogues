import numpy as np


class Higham(Exception):
    pass


def rq(A, x):
    """
    rg      Rayleigh quotient.
            rq(a, x) is the Rayleigh quotient of a and x, x'*A*x/(x'*x).

        Called by FV.
        NOTE: This function has a name clash with scipy.linalg.rq which
        computes the RQ decomposition of a matrix.
    """
    z = np.dot(x.T, np.dot(A, x)) / np.dot(x.T, x)

    return z.item()
