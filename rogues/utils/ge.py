import numpy as np


class Higham(Exception):
    pass


def ge(b):
    """
    GE     Gaussian elimination without pivoting.
       [l, u, rho] = ge(a) computes the factorization a = l*u,
       where L is unit lower triangular and U is upper triangular.
       RHO is the growth factor.
       By itself, ge(a) returns the final reduced matrix from the
       elimination containing both L and U.

       Note added in porting to Python/numpy/scipy:
       --------------------------------------------
       There are obviously more efficient routines in numpy / scipy
       but this routine is intended for testing numerical properties,
       ie it can be used in the direct search method adsmax for finding
       a matrix that maximizes the value of rho. See the following very
       readable and fun paper that is available from Prof. Higham's web
       site:

        Reference:
        N.J. Higham, Optimization by direct search in matrix computations,
        SIAM J. Matrix Anal. Appl, 14(2): 317-333, April 1993.

    """
    a = b.copy()    # don't cream the input matrix
    n, n = a.shape
    maxA = a.max()
    rho = maxA

    for k in range(n - 1):

        if a[k, k] == 0:
            raise Higham('Elimination breaks down with zero pivot.')

        a[k + 1:n, k] = a[k + 1:n, k] / a[k, k]          # Multipliers.

        # Elimination
        a[k + 1:n, k + 1:n] = (a[k + 1:n, k + 1:n] -
                               np.outer(a[k + 1:n, k], a[k, k + 1:n]))
        rho = max(rho, (abs(a[k + 1:n, k + 1:n])).max())

    l = np.tril(a, -1) + np.eye(n)
    u = np.triu(a)
    rho = rho / maxA

    return l, u, rho
