import numpy as np


class Higham(Exception):
    pass


def invhess(x, y=None):
    """
    invhess  inverse of an upper hessenberg matrix.
         invhess(x, y), where x is an n-vector and y an n-1 vector,
         is the matrix whose lower triangle agrees with that of
         np.outer(np.ones(n), x) and whose strict upper triangle agrees with
         that of np.outer(np.hstack((1, y)), np.ones(n))
         The matrix is nonsingular if x[1] != 0 and x[i+1] != y[i]
         for all i, and its inverse is an upper hessenberg matrix.
         If y is omitted it defaults to -x[0:n-2]
         Special case: if x is a scalar invhess(x) is the same as
         invhess(arange(1, x + 1)).

         References:
         F.N. Valvi and V.S. Geroyannis, Analytic inverses and
             determinants for a class of matrices, IMA Journal of Numerical
             Analysis, 7 (1987), pp. 123-128.
         W.-L. Cao and W.J. Stewart, A note on inverses of Hessenberg-like
             matrices, Linear Algebra and Appl., 76 (1986), pp. 233-240.
         Y. Ikebe, On inverses of Hessenberg matrices, Linear Algebra and
             Appl., 24 (1979), pp. 93-97.
         P. Rozsa, On the inverse of band matrices, Integral Equations and
             Operator Theory, 10 (1987), pp. 82-95.
    """
    try:
        n = np.max(x.shape)
    except AttributeError:
        n = x
        x = np.arange(1, n + 1)
        y = -1 * x

    a = np.outer(np.ones(n), x)
    for j in range(1, n):
        a[:j - 1, j] = y[:j - 1]

    return a
