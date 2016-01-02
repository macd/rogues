import numpy as np


class Higham(Exception):
    pass


def cond(a, p=2):
    """
    COND   Matrix condition number in 1, 2, Frobenius, or infinity norm.
       For p = 1, 2, 'fro', inf,  COND(A,p) = NORM(A,p) * NORM(INV(A),p).
       If p is omitted then p = 2 is used.
       A may be a rectangular matrix if p = 2; in this case COND(A)
       is the ratio of the largest singular value of A to the smallest
       (and hence is infinite if A is rank deficient).

       See also RCOND, NORM, CONDEST, NORMEST.

       This replicates (essentially) np.linalg.cond
    """

    if len(a) == 0:  # Handle null matrix.
        y = np.NaN
        return

    m, n = a.shape
    if m != n and p != 2:
        raise Higham('a is rectangular.  Use the 2 norm.')

    if p == 2:
        u, s, v = np.linalg.svd(a)
        if (s == 0).any():   # Handle singular matrix
            print('Condition is infinite')
            y = np.Inf
        y = max(s) / min(s)
    else:
        #  We'll let NORM pick up any invalid p argument.
        y = np.linalg.norm(a, p) * np.linalg.norm(np.linalg.inv(a), p)

    return y
