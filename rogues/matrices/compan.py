import numpy as np


def compan(p):
    """
    COMPAN  Companion matrix.
        compan(p) is a companion matrix.  There are three cases.
        If p is a scalar then compan(p) is the p-by-p matrix compan(1:p+1).
        If p is an (n+1)-vector, compan(p) is the n-by-n companion matrix
           whose first row is -p(2:n+1)/p(1).
        If p is a square matrix, compan(p) is the companion matrix
           of the characteristic polynomial of p, computed as
           compan(poly(p)).

        References:
        J.H. Wilkinson, The Algebraic Eigenvalue Problem,
           Oxford University Press, 1965, p. 12.
        G.H. Golub and C.F. Van Loan, Matrix Computations, second edition,
           Johns Hopkins University Press, Baltimore, Maryland, 1989,
           sec 7.4.6.
        C. Kenney and A.J. Laub, Controllability and stability radii for
          companion form systems, Math. Control Signals Systems, 1 (1988),
          pp. 239-256. (Gives explicit formulas for the singular values of
          COMPAN(P).)
    """
    try:
        n, m = p.shape
        if n == m and n > 1:
            # Matrix argument
            a = compan(np.poly(p))
            return a
        else:
            # Hmm, the commented stmt matches the logic in the m*lab code but
            # it would seem to be on a false path in the control logic,
            # ie compan shouldn't be defined for a non square matrix
            # so here we error out when that happens.
            # n = max(n,m)
            raise ValueError("Input matrix 'a' must be square "
                             "with dimension > 1")

    except ValueError:
        n, = p.shape

    except AttributeError:
        n = p + 1
        p = np.arange(1, n + 1)

    # Construct matrix of order n - 1
    if n == 2:
        a = 1
    else:
        a = np.diag(np.ones(n - 2), -1)
        a[0, :] = -p[1:n] / p[0]

    return a
