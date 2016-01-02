import numpy as np
from rogues.utils import toeplitz


def prolate(n, w=0.25):
    """
    PROLATE   Prolate matrix - symmetric, ill-conditioned Toeplitz matrix.
          A = PROLATE(N, W) is the N-by-N prolate matrix with parameter W.
          It is a symmetric Toeplitz matrix.
          If 0 < W < 0.5 then
             - A is positive definite
             - the eigenvalues of A are distinct, lie in (0, 1), and
               tend to cluster around 0 and 1.
          W defaults to 0.25.

          Reference:
          J.M. Varah. The Prolate matrix. Linear Algebra and Appl.,
          187:269--278, 1993.
    """
    a = np.zeros(n)
    a[0] = 2 * w
    a[1:n] = np.sin(2 * np.pi * w * np.arange(1, n)) / (np.pi *
                                                        np.arange(1, n))
    t = toeplitz(a)

    return t
