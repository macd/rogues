import numpy as np


def sub(a, i, j=None):
    """
    SUB     Principal submatrix.
        sub(a,i,j) is a[i:j, i:j].
        sub(a,i)  is the leading principal submatrix of order i,
        a[0:i, 0:i), if i > 0, and the trailing principal submatrix
        of order ABS(i) if i<0.
    """
    # editorial comment: this function seems unnecessary... it strikes
    # me that you would always want the bare code rather than the function
    # call because it is self evident what it does
    if j is None:
        if i >= 0:
            s = a[0:i, 0:i]
        else:
            n = np.min(a.shape)
            s = a[n + i:n, n + i:n]

    else:
        s = a[i:j, i:j]

    return s
