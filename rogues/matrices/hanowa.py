import numpy as np


class Higham(Exception):
    pass


def hanowa(n, d=1):
    """
    hanowa  a matrix whose eigenvalues lie on a vertical line in the
            complex plane.
        hanowa(n, d) is the n-by-n block 2x2 matrix (thus n = 2m must be even)
                      [d*np.eye(m)   -np.diag(np.range(1,m+1))
                       np.diag(arange(1,m+1))   d*np.eye(m) ]
        it has complex eigenvalues lambda(k) = d +/- k*i  (1 <= k <= m).
        parameter d defaults to -1.

        Reference:
        E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary
        Differential Equations I: Nonstiff Problems, Springer-Verlag,
        Berlin, 1987. (pp. 86-87)
    """
    if n % 2:
        raise Higham('n must be even.')
    else:
        m = n // 2

    dg = np.diag(np.arange(1, m + 1))
    # Style / lint checkers will complain about the following two variables
    # not being used because we are passing them into np.bmat as a string.
    mdg = -1 * dg
    de = d * np.eye(m)
    # Using this string form raises a deprecation error in numpy 2.1.0 and python 3.12.5
    a = np.bmat('de,  mdg; dg, de')

    return a
