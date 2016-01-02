import numpy as np
import rogues


def hadamard(n):
    """
    HADAMARD  Hadamard matrix.
          HADAMARD(N) is a Hadamard matrix of order N, that is,
          a matrix H with elements 1 or -1 such that H*H' = N*EYE(N).
          An N-by-N Hadamard matrix with N>2 exists only if REM(N,4) = 0.
          This function handles only the cases where N, N/12 or N/20
          is a power of 2.

          Reference:
          S.W. Golomb and L.D. Baumert, The search for Hadamard matrices,
             Amer. Math. Monthly, 70 (1963) pp. 12-17.
          http://en.wikipedia.org/wiki/Hadamard_matrix
          Weisstein, Eric W. "Hadamard Matrix." From MathWorld--
             A Wolfram Web Resource:
             http://mathworld.wolfram.com/HadamardMatrix.html
    """

    f, e = np.frexp(np.array([n, n / 12., n / 20.]))

    try:
        # If more than one condition is satified, this will always
        # pick the first one.
        k = [i for i in range(3) if (f == 0.5)[i] and (e > 0)[i]].pop()
    except IndexError:
        raise ValueError('N, N/12 or N/20 must be a power of 2.')

    e = e[k] - 1

    if k == 0:        # N = 1 * 2^e;
        h = np.array([1])

    elif k == 1:      # N = 12 * 2^e;
        tp = rogues.toeplitz(np.array([-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1]),
                             np.array([-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]))
        h = np.vstack((np.ones((1, 12)), np.hstack((np.ones((11, 1)), tp))))

    elif k == 2:     # N = 20 * 2^e;
        hk = rogues.hankel(
                np.array([-1, -1, 1, 1, -1, -1, -1, -1, 1,
                          -1, 1, -1, 1, 1, 1, 1, -1, -1, 1]),
                np.array([1, -1, -1, 1, 1, -1, -1, -1, -1,
                          1, -1, 1, -1, 1, 1, 1, 1, -1, -1]))
        h = np.vstack((np.ones((1, 20)), np.hstack((np.ones((19, 1)), hk))))

    #  Kronecker product construction.

    mh = -1 * h
    for i in range(e):
        ht = np.hstack((h, h))
        hb = np.hstack((h, mh))
        h = np.vstack((ht, hb))
        mh = -1 * h

    return h
