import numpy as np
from rogues.matrices import cauchy


def dingdong(n):
    """
    dingdong  dingdong matrix - a symmetric Hankel matrix.
          a = dingdong(n) is the symmetric n-by-n Hankel matrix with
                         a(i,j) = 0.5/(n-i-j+1.5).
          the eigenvalues of a cluster around pi/2 and -pi/2.

          Invented by F.N. Ris.

          Reference:
          J.C. Nash, Compact Numerical Methods for Computers: Linear
          Algebra and Function Minimisation, second edition, Adam Hilger,
          Bristol, 1990 (Appendix 1).
    """
    p = -2 * np.arange(1, n + 1) + (n + 1.5)
    a = cauchy(p)

    return a
