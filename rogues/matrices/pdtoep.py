import numpy as np


class Higham(Exception):
    pass


def pdtoep(n, m=None, w=None, theta=None):
    """
    PDTOEP   Symmetric positive definite Toeplitz matrix.
         PDTOEP(N, M, W, THETA) is an N-by-N symmetric positive (semi-)
         definite (SPD) Toeplitz matrix, comprised of the sum of M rank 2
         (or, for certain THETA, rank 1) SPD Toeplitz matrices.
         Specifically,
                 T = W(1)*T(THETA(1)) + ... + W(M)*T(THETA(M)),
         where T(THETA(k)) has (i,j) element COS(2*PI*THETA(k)*(i-j)).
         Defaults: M = N, W = RAND(M,1), THETA = RAND(M,1).

         Reference:
         G. Cybenko and C.F. Van Loan, Computing the minimum eigenvalue of
         a symmetric positive definite Toeplitz matrix, SIAM J. Sci. Stat.
         Comput., 7 (1986), pp. 123-131.
    """
    if m is None:
        m = n

    if w is None:
        w = np.random.rand(m)

    if theta is None:
        theta = np.random.rand(m)

    try:
        if np.max(w.shape) != m or np.max(theta.shape) != m:
            raise Higham('Arguments w and theta must be vectors of length M.')
    except:
        raise Higham('Arguments w and theta must be vectors of length M.')

    t = np.zeros(n)
    e = 2 * np.pi * (np.outer(np.arange(1, n + 1), np.ones(n)) -
                     np.ones(n) * np.arange(1, n + 1))

    for i in range(m):
        t = t + w[i] * np.cos(theta[i] * e)

    return t
