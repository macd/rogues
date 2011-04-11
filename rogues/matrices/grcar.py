import numpy as np


def grcar(n, k=3):
    """
    GRCAR     Grcar matrix - a Toeplitz matrix with sensitive eigenvalues.
          GRCAR(N, K) is an N-by-N matrix with -1s on the
          subdiagonal, 1s on the diagonal, and K superdiagonals of 1s.
          The default is K = 3.  The eigenvalues of this matrix form an
          interesting pattern in the complex plane (try ps(grcar(32))).
           or    a   = grcar(100)
                 w,v = np.eig(a)
                 pl.plot(w.real(), w.imag())

          References:
          J.F. Grcar, Operator coefficient methods for linear equations,
               Report SAND89-8691, Sandia National Laboratories, Albuquerque,
               New Mexico, 1989 (Appendix 2).
          N.M. Nachtigal, L. Reichel and L.N. Trefethen, A hybrid GMRES
               algorithm for nonsymmetric linear systems, SIAM J. Matrix Anal.
               Appl., 13 (1992), pp. 796-825.
    """

    g = np.tril(np.triu(np.ones((n, n))), k) - np.diag(np.ones(n - 1), -1)

    return g
