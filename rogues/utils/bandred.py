import numpy as np
from rogues.utils.house import house


class Higham(Exception):
    pass


def bandred(x, kl=0, ku=-1):
    """
    BANDRED  Band reduction by two-sided unitary transformations.
         B = BANDRED(A, KL, KU) is a matrix unitarily equivalent to A
         with lower bandwidth KL and upper bandwidth KU
         (i.e. B(i,j) = 0 if i > j + KL or j > i + KU).
         The reduction is performed using Householder transformations.
         If KU is omitted it defaults to KL.

         Called by RANDSVD.
         This is a `standard' reduction.  Cf. reduction to bidiagonal form
         prior to computing the SVD.  This code is a little wasteful in that
         it computes certain elements which are immediately set to zero!

         Reference:
         G.H. Golub and C.F. Van Loan, Matrix Computations, second edition,
         Johns Hopkins University Press, Baltimore, Maryland, 1989.
         Section 5.4.3.
    """
    # Work on a local copy of the array.
    a = x.copy()

    if kl == 0 and (ku == 0 or ku == -1):
        raise Higham("You've asked for a diagonal matrix. "
                     "In that case use the SVD!")
    elif ku == -1:
        # set ku to k1 if it has not yet been set
        ku = kl

    # Check for special case where order of left/right transformations matters.
    # Easiest approach is to work on the transpose, flipping back at the end.
    flip = 0
    if ku == 0:
        a = a.T
        kl, ku = ku, kl
        flip = 1

    m, n = a.shape

    for j in range(min(min(m, n), max(m - kl - 1, n - ku - 1))):
        if j + kl + 1 <= m:
            v, beta, s = house(a[j + kl:m, j])
            temp = a[j + kl:m, j:n]
            a[j + kl:m, j:n] = temp - beta * np.outer(v, np.dot(v, temp))
            a[j + kl + 1:m, j] = np.zeros(m - j - kl - 1)

        if j + ku + 1 <= n:
            v, beta, s = house(a[j, j + ku:n])
            temp = a[j:m, j + ku:n]
            a[j:m, j + ku:n] = temp - beta * np.outer(np.dot(temp, v), v)
            a[j, j + ku + 1:n] = np.zeros(n - j - ku - 1)

    if flip == 1:
        a = a.T

    return a
