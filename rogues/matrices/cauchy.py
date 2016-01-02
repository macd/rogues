import numpy as np


def cauchy(x, y=None):
    """
    cauchy  cauchy matrix.
        c = cauchy(x, y), where x, y are n-vectors, is the n-by-n matrix
        with c(i,j) = 1/(x(i)+y(j)).   By default, y = x.
        Special case: if x is a scalar cauchy(x) is the same as cauchy(1:x).
        Explicit formulas are known for DET(C) (which is nonzero if X and Y
        both have distinct elements) and the elements of INV(C).
        C is totally positive if 0 < X(1) < ... < X(N) and
        0 < Y(1) < ... < Y(N).

        References:
        N.J. Higham, Accuracy and Stability of Numerical Algorithms,
           Society for Industrial and Applied Mathematics, Philadelphia, PA,
           USA, 2002; sec. 28.1.
        D.E. Knuth, The Art of Computer Programming, Volume 1,
           Fundamental Algorithms, second edition, Addison-Wesley, Reading,
           Massachusetts, 1973, p. 36.
        E.E. Tyrtyshnikov, Cauchy-Toeplitz matrices and some applications,
           Linear Algebra and Appl., 149 (1991), pp. 1-18.
        O. Taussky and M. Marcus, Eigenvalues of finite matrices, in
           Survey of Numerical Analysis, J. Todd, ed., McGraw-Hill, New York,
           pp. 279-313, 1962. (States the totally positive property on p. 295.)
    """

    if not hasattr(x, 'shape'):
        x = np.arange(1, x)

    if y is None:
        y = x

    if not x.shape == y.shape:
        raise ValueError('Parameter vectors must be of same dimension.')

    n = x.shape[0]
    c = np.outer(x, np.ones(n)) + np.outer(np.ones(n), y)
    return 1. / c
