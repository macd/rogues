import numpy as np


def kms(n, rho=0.5):
    """
    kms   Kac-Murdock-Szego Toeplitz matrix.
      a = kms(n, rho) is the n-by-n Kac-Murdock-Szego Toeplitz matrix with
      a(i,j) = rho**(abs((i-j))) (for real rho).
      If RHO is complex, then the same formula holds except that elements
      below the diagonal are conjugated.
      RHO defaults to 0.5.
      Properties:

         a has an LDL' factorization with
                  L = INV(TRIW(N,-RHO,1)'),
                  D(i,i) = (1-ABS(RHO)^2)*EYE(N) except D(1,1) = 1.
            NOTE: I have not been able to verify this property. See
            test_rogues.py for more details.

         a is positive definite if and only if 0 < ABS(RHO) < 1.

         INV(A) is tridiagonal.
            NOTE: verified on several examples and used in unit test.

       Reference:
       W.F. Trench, Numerical solution of the eigenvalue problem
       for Hermitian Toeplitz matrices, SIAM J. Matrix Analysis and Appl.,
       10 (1989), pp. 135-146 (and see the references therein).
    """
    a = np.outer(np.arange(1, n + 1), np.ones(n))
    a = np.abs(a - a.T)
    a = rho ** a
    if np.iscomplex(rho):
        a = np.conj(np.tril(a, -1)) + np.triu(a)

    return a
