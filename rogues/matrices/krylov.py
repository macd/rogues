import numpy as np
import numpy.random as nrnd


def krylov(a, x=None, j=None):
    """
    KRYLOV    Krylov matrix.
          krylov(a, x, j) is the Krylov matrix
          [x, ax, a^2x, ..., a^(j-1)x],
          where a is an n-by-n matrix and x is an n-vector.
          defaults: x = np.ones(n), j = n.
          krylov(n) is the same as krylov(randn(n)).

          Reference:
          G.H. Golub and C.F. Van Loan, Matrix Computations, second edition,
          Johns Hopkins University Press, Baltimore, Maryland, 1989, p. 369.
    """
    try:
        n, n = a.shape
    except AttributeError:
        n = a
        a = nrnd.randn(n, n)

    if j is None:
        j = n

    if x is None:
        x = np.ones(n)

    b = np.ones((n, j))
    b[:, 0] = x

    for i in range(1, j):
        b[:, i] = np.dot(a, b[:, i - 1])

    return b
