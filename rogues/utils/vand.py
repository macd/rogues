import numpy as np


def vand(p, m=0):
    """
    VAND   Vandermonde matrix.
       v = vand(p), where p is a vector, produces the (primal)
       Vandermonde matrix based on the points p, i.e. v[i,j] = p[j]**(i-1)
       vand(p, m) is a rectangular version of vand(p) with m rows
       Special case: If P is a scalar then P equally spaced points on [0,1]
                     are used.

       Reference:
       N. J. Higham, Accuracy and Stability of Numerical Algorithms,
       Second edition, Society for Industrial and Applied Mathematics,
       Philadelphia, PA, 2002; chap. 22

       Notes added in porting:
       You should probably use np.vander instead of this function, but
       then the interface is different and you get the row reversed
       transpose of the matrix ( ie a.T[::-1,:] ) returned by vand

       WARNING... changed the function signiture from the Higham m*lab version
       by reversing the order of the arguments.
    """
    try:
        n = len(p)
    except TypeError:
        #  Handle scalar p.
        n = p
        p = np.linspace(0, 1, n)

    if m == 0:
        m = n

    v = np.ones((m, n))
    for i in range(1, m):
        v[i, :] = p * v[i - 1, :]

    return v
