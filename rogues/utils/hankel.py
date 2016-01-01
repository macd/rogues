import numpy as np


class Higham(Exception):
    pass


def hankel(a, b=None):
    """
    hankel(a) returns a toeplitz matrix given "a", the first row of the
    matrix.  This matrix is defined as:

         [[   a[0], a[1], a[2], a[3], ..., a[n-1], a[n] ]]
          [   a[1], a[2], a[3], a[4], ...,   a[n],   0  ]
          [   a[2], a[3], a[4], ...,     ]
          ...
          [ a[n-2], a[n-1], a[n],  0,   ...
          [ a[n-1], a[n],      0,  0,   ...           0
          [   a[n],    0,      0,  0,   ...           0 ] ]


    Note that all the non-zero anti-diagonals are constant

    If called as hankel(a, b) then create the hankel matrix where a is the
    first column and b is the last row.  If a[-1] != b[0], a[-1] is chosen
    but a warning message is printed.  For example

    In [2]: from hankel import *
    In [3]: a = arange(10)
    In [4]: a
    Out[4]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    In [5]: b = arange(10,20)
    In [6]: h = hankel(a, b)
    Warning: a[-1] != b[0]. a[-1] is chosen
    In [7]: h
    Out[7]:
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
           [ 2,  3,  4,  5,  6,  7,  8,  9, 11, 12],
           [ 3,  4,  5,  6,  7,  8,  9, 11, 12, 13],
           [ 4,  5,  6,  7,  8,  9, 11, 12, 13, 14],

           [ 5,  6,  7,  8,  9, 11, 12, 13, 14, 15],
           [ 6,  7,  8,  9, 11, 12, 13, 14, 15, 16],
           [ 7,  8,  9, 11, 12, 13, 14, 15, 16, 17],
           [ 8,  9, 11, 12, 13, 14, 15, 16, 17, 18],
           [ 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]])

    NOTE: This looks like a duplicate from rogues/matrics/hankel.py
    """

    # Error checking...
    try:
        m, = a.shape
        if b is not None:
            n, = b.shape
    except (ValueError, AttributeError):
        raise Higham("Input arrays must be one dimensional")

    if b is None:
        b = np.zeros_like(a)
        n = m
    elif a[-1] != b[0]:
        print("Warning: a[-1] != b[0]. a[-1] is chosen")

    k = np.hstack((a, b[1:]))
    h = k[:n]
    for i in range(1, m):
        h = np.vstack((h, k[i:i + n]))

    return h
