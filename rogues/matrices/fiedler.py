import numpy as np


def fiedler(c):
    """
    FIEDLER  Fiedler matrix - symmetric.
         A = FIEDLER(C), where C is an n-vector, is the n-by-n symmetric
         matrix with elements ABS(C(i)-C(j)).
         Special case: if C is a scalar, then A = FIEDLER(1:C)
                       (i.e. A(i,j) = ABS(i-j)).
         Properties:
           FIEDLER(N) has a dominant positive eigenvalue and all the other
                      eigenvalues are negative (Szego, 1936).
           Explicit formulas for INV(A) and DET(A) are given by Todd (1977)
           and attributed to Fiedler.  These indicate that INV(A) is
           tridiagonal except for nonzero (1,n) and (n,1) elements.
           [I think these formulas are valid only if the elements of
           C are in increasing or decreasing order---NJH.]

           References:
           G. Szego, Solution to problem 3705, Amer. Math. Monthly,
              43 (1936), pp. 246-259.
           J. Todd, Basic Numerical Mathematics, Vol. 2: Numerical Algebra,
              Birkhauser, Basel, and Academic Press, New York, 1977, p. 159.
    """

    try:
        if len(c.shape) != 1:
            raise ValueError("Only 1-D vectors or scalers are valid input")
        n, = c.shape

    except AttributeError:
        # c must be a scalar integer
        n = c
        c = np.arange(1, n + 1)

    a = np.ones((n, n)) * c
    a = abs(a - a.T)        # NB. array transpose.

    return a
