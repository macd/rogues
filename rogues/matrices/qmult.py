import numpy as np
import numpy.linalg as nl
import numpy.random as nrnd


def qmult(b):
    """
    QMULT  Pre-multiply by random orthogonal matrix.
       QMULT(A) is Q*A where Q is a random real orthogonal matrix from
       the Haar distribution, of dimension the number of rows in A.
       Special case: if A is a scalar then QMULT(A) is the same as
                     QMULT(EYE(A)).

       Called by RANDSVD.

       Reference:
       G.W. Stewart, The efficient generation of random
       orthogonal matrices with an application to condition estimators,
       SIAM J. Numer. Anal., 17 (1980), 403-409.
    """
    try:
        n, m = b.shape
        a = b.copy()

    except AttributeError:
        n = b
        a = np.eye(n)

    d = np.zeros(n)

    for k in range(n - 2, -1, -1):
        # Generate random Householder transformation.
        x = nrnd.randn(n - k)
        s = nl.norm(x)
        # Modification to make sign(0) == 1
        sgn = np.sign(x[0]) + float(x[0] == 0)
        s = sgn * s
        d[k] = -sgn
        x[0] = x[0] + s
        beta = s * x[0]

        # Apply the transformation to a
        y = np.dot(x, a[k:n, :])
        a[k:n, :] = a[k:n, :] - np.outer(x, (y / beta))

    # Tidy up signs.
    for i in range(n - 1):
        a[i, :] = d[i] * a[i, :]

    # Now randomly change the sign (Gaussian dist)
    a[n - 1, :] = a[n - 1, :] * np.sign(nrnd.randn())

    return a
