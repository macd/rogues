import numpy as np


def lehmer(n):
    """
    lehmer  lehmer matrix - symmetric positive definite.
        a = lehmer(n) is the symmetric positive definite n-by-n matrix with
                         a[i,j] = (i+1)/(j+1) for j >= i.
        a is totally nonnegative.  inv(a) is tridiagonal, and explicit
        formulas are known for its entries.
        n <= cond(a) <= 4*n*n.

        References:
        M. Newman and J. Todd, The evaluation of matrix inversion
           programs, J. Soc. Indust. Appl. Math., 6 (1958), pp. 466-476.
        Solutions to problem E710 (proposed by D.H. Lehmer): The inverse
           of a matrix, Amer. Math. Monthly, 53 (1946), pp. 534-535.
        J. Todd, Basic Numerical Mathematics, Vol. 2: Numerical Algebra,
           Birkhauser, Basel, and Academic Press, New York, 1977, p. 154.
    """
    a = np.outer(np.ones(n), np.arange(1, n + 1))
    a = a / a.T
    a = np.tril(a) + np.tril(a, -1).T

    return a
