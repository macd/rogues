import numpy as np


def jordbloc(n, lambduh=1):
    """
    jordbloc  jordan block.
          jordbloc(n, lambda) is the n-by-n jordan block with eigenvalue
          lambduh.  lambduh = 1 is the default. (Recall that lambda is a
          reserved word in Python)
    """

    j = lambduh * np.eye(n) + np.diag(np.ones(n - 1), 1)

    return j
