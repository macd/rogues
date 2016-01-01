import numpy as np


def chebvand(m, p=None):
    """
    CHEBVAND  Vandermonde-like matrix for the Chebyshev polynomials.
          c = chebvand(p), where P is a vector, produces the (primal)
          Chebyshev Vandermonde matrix based on the points P,
          i.e., c[i,j] = t_{i-1}(p[j]), where T_{i-1} is the Chebyshev
          polynomial of degree i-1.
          CHEBVAND(M,P) is a rectangular version of CHEBVAND(P) with M rows.
          Special case: If P is a scalar then P equally spaced points on
                        [0,1] are used.

           Reference:
           N.J. Higham, Stability analysis of algorithms for solving confluent
           Vandermonde-like systems, SIAM J. Matrix Anal. Appl., 11 (1990),
           pp. 23-41.
    """
    if p is None:
        nargin = 1
        p = m
    else:
        nargin = 2

    try:
        n = np.max(p.shape)
    except AttributeError:
        n = p
        p = np.linspace(0., 1., n)

    if nargin == 1:
        m = n

    c = np.ones((m, n))
    if m == 1:
        return c

    c[1, :] = p

    #  Use Chebyshev polynomial recurrence.
    for i in range(2, m):
        c[i, :] = 2. * p * c[i - 1, :] - c[i - 2, :]

    return c
