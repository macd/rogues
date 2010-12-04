import numpy as np

class Higham(Exception):
    pass

def cauchy(x, y = None, overwrite_a = False):
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
    try:
        n, = x.shape
    except AttributeError:
        n = x
        a = np.arange(1, n + 1)
        
    if y == None:
        y = x

    if not x.shape == y.shape:
        raise Higham('Parameter vectors must be of same dimension.')

    c = np.outer(x, np.ones(n)) + np.outer(np.ones(n), y)
    c = 1 / c
    return c
