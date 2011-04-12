import numpy as np
import numpy.random as nrnd


class Higham(Exception):
    pass


def sparsify(a, p=0.25):
    """
    SPARSIFY  Randomly set matrix elements to zero.
          S = SPARSIFY(A, P) is A with elements randomly set to zero
          (S = S' if A is square and A = A', i.e. symmetry is preserved).
          Each element has probability P of being zeroed.
          Thus on average 100*P percent of the elements of A will be zeroed.
          Default: P = 0.25.

          Note added in porting: by inspection only, it appears the the m*lab
          version may have a bug where it always returns zeros on the diagonal
          for a symmetric matrix... can anyone confirm?
     """

    if p < 0 or p > 1:
        raise Higham('Second parameter must be between 0 and 1 inclusive.')

    m, n = a.shape

    if (a == a.T).all():
        # Preserve symmetry
        d = np.choose(nrnd.rand(m) > p, (np.zeros(m), np.diag(a)))
        a = np.triu(a, 1) * (nrnd.rand(m, n) > p)
        a = a + a.T
        a = a + np.diag(d)
    else:
        # Unsymmetric case
        a = np.choose(nrnd.rand(m, n) > p, (np.zeros((m, n)), a))

    return a
