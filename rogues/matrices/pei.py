import numpy as np


def pei(n, alpha=1):
    """
    PEI    Pei matrix.
       pei(n, alpha), where alpha is a scalar, is the symmetric matrix
       alpha*eye(n) + ones((n,n)).
       If alpha is omitted then alpha = 1 is used.
       The matrix is singular for ALPHA = 0, -N.

       Reference:
       M.L. Pei, A test matrix for inversion procedures,
       Comm. ACM, 5 (1962), p. 508.
    """
    p = alpha * np.eye(n) + 1.

    return p
