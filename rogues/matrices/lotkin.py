import numpy as np
from rogues.matrices import hilb


def lotkin(n):
    """
    lotkin  lotkin matrix.
        a = lotkin(n) is the Hilbert matrix with its first row altered to
        all ones.  A is unsymmetric, ill-conditioned, and has many negative
        eigenvalues of small magnitude.
        The inverse has integer entries and is known explicitly.

        Reference:
        M. Lotkin, A set of test matrices, MTAC, 9 (1955), pp. 153-161.
    """
    a = hilb(n)
    a[0, :] = np.ones(n)

    return a
