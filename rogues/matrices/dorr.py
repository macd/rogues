import numpy as np
import rogues


def dorr(n, theta=0.01, r_matrix=False, set_sparse=False):
    """
    dorr  dorr matrix - diagonally dominant, ill conditioned, tridiagonal.
      c, d, e = dorr(n, theta) returns the vectors defining a row diagonally
      dominant, tridiagonal m-matrix that is ill conditioned for small
      values of the parameter theta >= 0.
      If r_matrix is set to True, then c = full(tridiag(c,d,e)), i.e.,
      the matrix iself is returned.  The format for the matrix is set by
      the boolean set_sparse.  If set to true, return a matrix in sparse
      matrix format.  Default is False.

      The columns of inv(c) vary greatly in norm.  theta defaults to 0.01.
      The amount of diagonal dominance is given by (ignoring rounding errors):
            comp(c)*ones(n,1) = theta*(n+1)^2 * [1 0 0 ... 0 1]'.

      Reference:
      F.W. Dorr, An example of ill-conditioning in the numerical
      solution of singular perturbation problems, Math. Comp., 25 (1971),
      pp. 271-283.
    """

    c = np.zeros(n)
    e = np.zeros(n)
    d = np.zeros(n)

    # All length n for convenience.  Make c, e of length n-1 later.

    h = 1. / (n + 1)
    m = (n + 1) // 2
    term = theta / h ** 2

    i = slice(0, m)
    c[i] = -term * np.ones(m)
    e[i] = c[i] - (0.5 - np.arange(1, m + 1) * h) / h
    d[i] = -(c[i] + e[i])

    i = slice(m, n)
    e[i] = -term * np.ones(n - m)
    c[i] = e[i] + (0.5 - np.arange(1, m + 1) * h) / h
    d[i] = -(c[i] + e[i])

    # shorten
    c = c[1:n]
    e = e[0:n - 1]

    if r_matrix:
        if set_sparse:
            c = rogues.tridiag(c, d, e)
        else:
            c = np.diag(c, -1) + np.diag(d) + np.diag(e, 1)
        # if we are to return the matrix, return _only_ the matrix
        return c

    # Return the diagonals of the matrix
    return c, d, e
