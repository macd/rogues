import numpy as np
from rogues.utils import tridiag

class Higham(Exception):
    pass

def neumann(n):
    """
    neumann  Singular matrix from the discrete Neumann problem (sparse).
         neumann(n) is the singular, row diagonally dominant matrix resulting
         from discretizing the neumann problem with the usual five point
         operator on a regular mesh.
         It has a one-dimensional null space with null vector ones(n,1).
         the dimension n should be a perfect square, or else a 2-vector,
         in which case the dimension of the matrix is n[1]*n[2].

         Reference:
         R.J. Plemmons, Regular splittings and the discrete Neumann
         problem, Numer. Math., 25 (1976), pp. 153-161.
    """
    try:
        m, n = n.shape
    except AttributeError:
        m = int(np.sqrt(n))
        if m**2 != n:
            raise Higham('N must be a perfect square.')
        n = m

    t = tridiag(m, -1, 2, -1).todense()
    t[0, 1] = -2
    t[m - 1, m - 2] = -2

    a = np.kron(t, np.eye(n)) + np.kron(np.eye(n), t)

    return a, t
