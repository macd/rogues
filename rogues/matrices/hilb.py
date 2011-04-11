import numpy as np


class Higham(Exception):
    pass


def hilb(n, m=0):
    """
    hilb   Hilbert matrix.
       hilb(n,m) is the n-by-m matrix with elements 1/(i+j-1).
       it is a famous example of a badly conditioned matrix.
       cond(hilb(n)) grows like exp(3.5*n).
       hilb(n) is symmetric positive definite, totally positive, and a
       Hankel matrix.

       References:
       M.-D. Choi, Tricks or treats with the Hilbert matrix, Amer. Math.
           Monthly, 90 (1983), pp. 301-312.
       N.J. Higham, Accuracy and Stability of Numerical Algorithms,
           Society for Industrial and Applied Mathematics, Philadelphia, PA,
           USA, 2002; sec. 28.1.
       M. Newman and J. Todd, The evaluation of matrix inversion
           programs, J. Soc. Indust. Appl. Math., 6 (1958), pp. 466-476.
       D.E. Knuth, The Art of Computer Programming,
           Volume 1, Fundamental Algorithms, second edition, Addison-Wesley,
           Reading, Massachusetts, 1973, p. 37.

       NOTE added in porting.  We do not use the function cauchy here to
       generate the Hilbert matrix.  That is done so we can unit test the
       the functions against each other.  Also, the function has been
       generalized to take by row and column sizes.  If only a row size
       is given, we assume a square matrix is desired.
    """
    if n < 1 or m < 0:
        raise Higham("Matrix size must be one or greater")
    elif n == 1 and (m == 0 or m == 1):
        return np.array([[1]])
    elif m == 0:
        m = n

    v = np.arange(1, n + 1) + np.arange(0, m)[:, np.newaxis]
    return 1. / v
