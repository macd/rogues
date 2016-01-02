import numpy as np


def smoke(n, k=0):
    """
    smoke     smoke matrix - complex, with a `smoke ring' pseudospectrum.
          smoke(n) is an n-by-n matrix with 1s on the
          superdiagonal, 1 in the (n,1) position, and powers of
          roots of unity along the diagonal.
          smoke(n, 1) is the same except for a zero (n,1) element.
          the eigenvalues of smoke(n, 1) are the n'th roots of unity;
          those of smoke(n) are the n'th roots of unity times 2^(1/n).

          Try ps(smoke(32)).  For smoke(n, 1) the pseudospectrum looks
          like a sausage folded back on itself.
          gersh(smoke(n, 1)) is interesting.

          Reference:
          L. Reichel and L.N. Trefethen, Eigenvalues and pseudo-eigenvalues of
          Toeplitz matrices, Linear Algebra and Appl., 162-164:153-185, 1992.
    """
    w = np.exp(2 * np.pi * 1j / n)
    a = (np.diag(np.hstack((w ** np.arange(1, n), 1))) +
         np.diag(np.ones(n - 1), 1))
    if k == 0:
        a[n - 1, 0] = 1

    return a
