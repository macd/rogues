import numpy as np


def clement(n, k=0):
    """
    CLEMENT   Clement matrix - tridiagonal with zero diagonal entries.
          CLEMENT(N, K) is a tridiagonal matrix with zero diagonal entries
          and known eigenvalues.  It is singular if N is odd.  About 64
          percent of the entries of the inverse are zero.  The eigenvalues
          are plus and minus the numbers N-1, N-3, N-5, ..., (1 or 0).
          For K = 0 (the default) the matrix is unsymmetric, while for
          K = 1 it is symmetric.
          CLEMENT(N, 1) is diagonally similar to CLEMENT(N).

          Similar properties hold for TRIDIAG(X,Y,Z) where Y = ZEROS(N,1).
          The eigenvalues still come in plus/minus pairs but they are not
          known explicitly.

          References:
          P.A. Clement, A class of triple-diagonal matrices for test
             purposes, SIAM Review, 1 (1959), pp. 50-52.
          A. Edelman and E. Kostlan, The road from Kac's matrix to Kac's
             random polynomials. In John~G. Lewis, editor, Proceedings of
             the Fifth SIAM Conference on Applied Linear Algebra Society
             for Industrial and Applied Mathematics, Philadelphia, 1994,
             pp. 503-507.
          O. Taussky and J. Todd, Another look at a matrix of Mark Kac,
             Linear Algebra and Appl., 150 (1991), pp. 341-360.
    """
    z = np.arange(1, n)

    if k == 0:
        a = np.diag(z[::-1], -1) + np.diag(z, 1)
    else:
        y = np.sqrt(z[::-1] * z)
        a = np.diag(y, -1) + np.diag(y, 1)

    return a
