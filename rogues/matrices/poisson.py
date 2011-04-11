import scipy.sparse as sparse
from rogues.utils import tridiag


def poisson(n):
    """
    poisson   Block tridiagonal matrix from Poisson's equation (sparse).
          poisson(n) is the block tridiagonal matrix of order n**2
          resulting from discretizing Poisson's equation with the
          5-point operator on an n-by-n mesh.

          Reference:
          G.H. Golub and C.F. Van Loan, Matrix Computations, second edition,
          Johns Hopkins University Press, Baltimore, Maryland, 1989
          (Section 4.5.4).
    """
    s = tridiag(n, -1, 2, -1)
    i = sparse.eye(n, n)
    a = sparse.kron(i, s) + sparse.kron(s, i)

    return a
