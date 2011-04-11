import numpy as np


def rando(n, k=1):
    """
    RANDO   Random matrix with elements -1, 0 or 1.
        a = rando(n, k) is a random n-by-n matrix with elements from
        one of the following discrete distributions (default k = 1):
          k = 1:  a[i,j] =  0 or 1    with equal probability,
          k = 2:  a[i,j] = -1 or 1    with equal probability,
          k = 3:  a[i,j] = -1, 0 or 1 with equal probability.
        n may be a 2-tuple, in which case the matrix is n[0]-by-n[1].
    """
    try:
        m, n = n                  # Parameter n specifies dimension: m-by-n.
    except TypeError:
        m = n

    if k == 1:                    # {0, 1}
        a = np.floor(np.random.rand(m, n) + .5)

    elif k == 2:                  # {-1, 1}
        a = 2 * np.floor(np.random.rand(m, n) + .5) - 1

    elif k == 3:                  # {-1, 0, 1}
        a = np.round(3 * np.random.rand(m, n) - 1.5)

    return a
