import numpy as np
from rogues.matrices.triw import triw


def moler(n, alpha=-1):
    """
    MOLER   Moler matrix - symmetric positive definite.
        A = MOLER(N, ALPHA) is the symmetric positive definite N-by-N matrix
        U'*U where U = TRIW(N, ALPHA).
        For ALPHA = -1 (the default) A(i,j) = MIN(i,j)-2, A(i,i) = i.
        A has one small eigenvalue.

        Nash (1990) attributes the ALPHA = -1 matrix to Moler.

        Reference:
        J.C. Nash, Compact Numerical Methods for Computers: Linear
        Algebra and Function Minimisation, second edition, Adam Hilger,
        Bristol, 1990 (Appendix 1).
    """
    a = triw(n, alpha).T  @  triw(n, alpha)
    return a
