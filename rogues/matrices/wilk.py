import numpy as np
from rogues.matrices import hilb


def wilk(n):
    """
    wilk   Various specific matrices devised/discussed by Wilkinson.
       a, b = wilk(n) is the matrix or system of order N.
       N = 3: upper triangular system Ux=b illustrating inaccurate solution.
       N = 4: lower triangular system Lx=b, ill-conditioned.
       N = 5: HILB(6)(1:5,2:6)*1.8144.  Symmetric positive definite.
       N = 21: W21+, tridiagonal.   Eigenvalue problem.

       References:
       J.H. Wilkinson, Error analysis of direct methods of matrix inversion,
          J. Assoc. Comput. Mach., 8 (1961),  pp. 281-330.
       J.H. Wilkinson, Rounding Errors in Algebraic Processes, Notes on Applied
          Science No. 32, Her Majesty's Stationery Office, London, 1963.
       J.H. Wilkinson, The Algebraic Eigenvalue Problem, Oxford University
          Press, 1965.
    """
    b = []
    if n == 3:
        # Wilkinson (1961) p.323.
        a = [[1e-10,   .9,  -.4],
             [0,     .9,  -.4],
             [0,     0,  1e-10]]
        b = [0,      0,    1]

    elif n == 4:
        # Wilkinson (1963) p.105.
        a = [[0.9143e-4, 0, 0, 0],
             [0.8762, 0.7156e-4, 0, 0],
             [0.7943, 0.8143, 0.9504e-4, 0],
             [0.8017, 0.6123, 0.7165, 0.7123e-4]]
        b = [0.6524, 0.3127, 0.4186, 0.7853]

    elif n == 5:
        # Wilkinson (1965), p.234.
        a = hilb(6, 6)
        # drop off the last row and the first column
        a = a[0:5, 1:6] * 1.8144
        # return zero array for b
        b = np.zeros(5)

    elif n == 21:
        # Taken from gallery.m.  Wilkinson (1965), p.308.
        E = np.diag(np.ones(n - 1), 1)
        m = (n - 1) / 2
        a = np.diag(np.abs(np.arange(-m, m + 1))) + E + E.T
        # return zero array for b
        b = np.zeros(21)

    else:
        raise ValueError("Sorry, that value of N is not available.")

    return np.array(a), np.array(b)
