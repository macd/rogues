import numpy as np


def gfpp(t, c=1.0):
    """
    GFPP   Matrix giving maximal growth factor for Gaussian elim. with pivoting
       GFPP(T) is a matrix of order N for which Gaussian elimination
       with partial pivoting yields a growth factor 2^(N-1).
       T is an arbitrary nonsingular upper triangular matrix of order N-1.
       GFPP(T, C) sets all the multipliers to C  (0 <= C <= 1)
       and gives growth factor (1+C)^(N-1).
       GFPP(N, C) (a special case) is the same as GFPP(EYE(N-1), C) and
       generates the well-known example of Wilkinson.

       Reference:
       N.J. Higham and D.J. Higham, Large growth factors in
       Gaussian elimination with pivoting, SIAM J. Matrix Analysis and
       Appl., 10 (1989), pp. 155-164.
    """

    try:
        m, = t.shape
        # t must be an array rather than an int
        n = m + 1
        if np.linalg.norm(t - np.triu(t), 1) | (np.diag(t) == 0.0).any():
            raise ValueError('First argument must be a nonsingular upper'
                             ' triangular matrix.')
    except AttributeError:
        # Handle the special case T = scalar
        n = t
        m = n - 1
        t = np.eye(n - 1)

    if c < 0. or c > 1.:
        raise Higham('Second parameter must be a scalar between 0 and '
                     '1 inclusive.')

    d = 1. + c
    l = np.eye(n) - c * np.tril(np.ones((n, n)), -1)

    uu = np.hstack((t, (d ** np.arange(0, n - 1)).reshape(n - 1, 1)))
    ul = np.zeros((1, m + 1))
    ul[0, m] = d ** (n - 1)
    u = np.vstack((uu, ul))
    a = l @ u
    theta = np.abs(a).max()
    a[:, n - 1] = (theta / np.linalg.norm(a[:, n - 1], np.inf)) * a[:, n - 1]

    return np.asarray(a)
