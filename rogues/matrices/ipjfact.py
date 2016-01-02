import numpy as np
from rogues.matrices import hankel


def ipjfact(n, k=0):
    """
    ipjfact   A Hankel matrix with factorial elements.
          a = ipjfact(n, k) is the matrix with
                    a(i,j) = (i+j)!    (k = 0, default)
                    a(i,j) = 1/(i+j)!  (k = 1)
          both are hankel matrices.
          The determinant and inverse are known explicitly.
          d = det(a) is returned is always returned as in
          a, d = ipjfact(n, k)

          Suggested by P. R. Graves-Morris.

          Reference:
          M.J.C. Gover, The explicit inverse of factorial Hankel matrices,
          Dept. of Mathematics, University of Bradford, 1993.
    """
    c = np.cumprod(np.arange(2, n + 2))
    d = np.cumprod(np.arange(n + 1, 2 * n + 1)) * c[n - 2]

    a = hankel(c, d)

    if k == 1:
        a = 1 / a

    d = 1

    #
    # There appears to be a bug in the implementaton of interger
    # multiply in numpy (note _not_ in Python).  Therefore we use
    # the explicit "cast" to float64 below.
    #
    if k == 0:
        for i in range(1, n):
            d = d * np.prod(np.arange(1, i + 2, dtype='float64')) *  \
                np.prod(np.arange(1, n - i + 1, dtype='float64'))

        d = d * np.prod(np.arange(1, n + 2, dtype='float64'))
    else:
        for i in range(0, n):
            d = d * np.prod(np.arange(1, i, dtype='float64')) /    \
                np.prod(np.arange(1, n + 1 + i, dtype='float64'))

        if (n * (n - 1) / 2) % 2:
            d = -d
    det_a = d

    return a, det_a
