import numpy as np
from rogues.utils import tridiag


def lesp(n):
    """
    LESP   A tridiagonal matrix with real, sensitive eigenvalues.
       LESP(N) is an N-by-N matrix whose eigenvalues are real and smoothly
       distributed in the interval approximately [-2*N-3.5, -4.5].
       The sensitivities of the eigenvalues increase exponentially as
       the eigenvalues grow more negative.
       The matrix is similar to the symmetric tridiagonal matrix with
       the same diagonal entries and with off-diagonal entries 1,
       via a similarity transformation with d = np.diag(1!,2!,...,N!).

       References:
       H.W.J. Lenferink and M.N. Spijker, On the use of stability regions in
            the numerical analysis of initial value problems,
            Math. Comp., 57 (1991), pp. 221-237.
       L.N. Trefethen, Pseudospectra of matrices, in Numerical Analysis 1991,
            Proceedings of the 14th Dundee Conference,
            D.F. Griffiths and G.A. Watson, eds, Pitman Research Notes in
            Mathematics, volume 260, Longman Scientific and Technical, Essex,
            UK, 1992, pp. 234-266.
    """
    x = np.arange(2., n + 1)
    y = -(2. * np.arange(2, n + 2) + 1)
    t = tridiag(1 / x, y, x).todense()

    return t
