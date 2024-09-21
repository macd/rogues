import numpy as np


def kahan(n, theta=1.2, pert=25):
    """
    kahan  kahan matrix - upper trapezoidal.
       kahan(n, theta) is an upper trapezoidal matrix
       that has some interesting properties regarding estimation of
       condition and rank.
       The matrix is n-by-n unless n is a 2-tuple, in which case it
       is n[0]-by-n[1].
       The parameter theta defaults to 1.2.
       The useful range of theta is 0 < theta < pi.

       To ensure that the QR factorization with column pivoting does not
       interchange columns in the presence of rounding errors, the diagonal
       is perturbed by pert*eps*np.diag( arange(n,0,-1)).
       The default is pert = 25, which ensures no interchanges for kahan(n)
       up to at least n = 90 in IEEE arithmetic.
       kahan(n, theta, pert) uses the given value of pert.

       The inverse of kahan(n, theta) is known explicitly: see
       Higham (1987, p. 588), for example.
       The diagonal perturbation was suggested by Christian Bischof.

       References:
       W. Kahan, Numerical linear algebra, Canadian Math. Bulletin,
          9 (1966), pp. 757-801.
       N.J. Higham, A survey of condition number estimation for
          triangular matrices, SIAM Review, 29 (1987), pp. 575-596.
    """

    try:
        r, n = n              # Parameter n specifies dimension: r-by-n.
    except TypeError:
        r = n

    s = np.sin(theta)
    c = np.cos(theta)

    u = np.eye(n) - c * np.triu(np.ones((n, n)), 1)
    z = np.diag(s ** np.arange(0, n))
    eps = np.finfo(float).eps
    u = z @ u + pert * eps * np.diag(np.arange(n, 0, -1))

    if r > n:
        # Extend to an r-by-n matrix.
        u = np.vstack((u, np.zeros(((r - n), n))))
    elif r < n:
        # Reduce to an r-by-n matrix.
        u = u[:r, :]

    return u
