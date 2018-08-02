import numpy as np


class Higham(Exception):
    pass


def ohess(x):
    """
    ohess  random, orthogonal upper hessenberg matrix.
       h = ohess(n) is an n-by-n real, random, orthogonal
       upper Hessenberg matrix.
       Alternatively, H = OHESS(X), where X is an arbitrary real
       N-vector (N > 1) constructs H non-randomly using the elements
       of X as parameters.
       In both cases H is constructed via a product of N-1 Givens rotations.

       Note: See Gragg (1986) for how to represent an N-by-N (complex)
       unitary Hessenberg matrix with positive subdiagonal elements in terms
       of 2N-1 real parameters (the Schur parameterization).
       This py-file handles the real case only and is intended simply as a
       convenient way to generate random or non-random orthogonal Hessenberg
       matrices.

       Reference:
       W.B. Gragg, The QR algorithm for unitary Hessenberg matrices,
       J. Comp. Appl. Math., 16 (1986), pp. 1-8.
    """

    if type(x) == int:
        n = x
        x = np.random.uniform(size=n - 1) * 2 * np.pi
        h = np.eye(n)
        h[n - 1, n - 1] = np.sign(np.random.randn())
    elif type(x) == numpy.ndarray:
        if np.imag(x).any():
            raise Higham('Parameter must be real.')
        n = np.max(x.shape)
        h = np.eye(n)
        # Second term ensures h[n-1, n-1] nonzero.
        h[n - 1, n - 1] = np.sign(x[n - 1]) + float(x[n - 1] == 0)
    else:
        raise Higham('Unknown type in ohess')        
        
        
    for i in range(n - 1, 0, -1):
        # Apply Givens rotation through angle x[i - 1]
        theta = x[i - 1]
        c = np.cos(theta)
        s = np.sin(theta)
        h[i - 1:i + 1, :] = np.vstack((c * h[i - 1, :] + s * h[i, :],
                                       -s * h[i - 1, :] + c * h[i, :]))

    return h
