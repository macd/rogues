import numpy as np


class Higham(Exception):
    pass


def treshape(x, unit=0, row_wise=False):
    """
    treshape  reshape vector to or from (unit) triangular matrix.
          treshape(x) returns a square upper triangular matrix whose
          elements are taken columnwise from the matrix x.
          If row_wise = True, then the matrix elements are taken
          row wise from the matrix x.
          treshape(x,1) returns a UNIT upper triangular matrix, and
          the 1's should not be specified in X.
          An error results if X does not have a number of elements of the form
          n*(n+1)/2 (or n less than this in the unit triangular case).
          x = treshape(r,2) is the inverse operation to r = treshape(x).
          x = treshape(r,3) is the inverse operation to r = treshape(x,1).
    """
    try:
        p, q = x.shape
    except ValueError:
        p, = x.shape
        q = 1

    if unit < 2:   # Convert vector x to upper triangular R.
        m = p * q
        n = int(np.around((-1 + np.sqrt(1 + 8 * m)) / 2.))
        if n * (n + 1) // 2 != m:
            raise Higham('Matrix must have a "triangular" '
                         'number of elements.')

        if unit == 1:
            n = n + 1

        x = x.ravel()
        t = unit * np.eye(n)

        i = 0
        if row_wise:
            for j in range(n - unit):
                t[j, j + unit:n] = x[i:i + n - unit - j]
                i = i + n - unit - j

        else:
            for j in range(unit, n):
                t[0:j - unit + 1, j] = x[i:i + j - unit + 1]
                i = i + j - unit + 1

    elif unit >= 2:   # Convert upper triangular R to vector x.
        t = x
        if p != q:
            raise Higham('Must pass square matrix')

        unit = unit - 2
        n = p * (p + 1) // 2 - unit * p
        x = np.zeros(n)
        i = 0
        if row_wise:
            for j in range(p - unit):
                x[i:i + p - unit - j] = t[j, j + unit:p]
                i = i + p - unit - j

        else:
            for j in range(unit, p):
                x[i:i + j - unit + 1] = t[0:j - unit + 1, j]
                i = i + j - unit + 1

        t = x

    return t
