import numpy as np


def comp(a, k=0):
    """
    COMP    Comparison matrices.
        comp(a) is diag(b) - tril(b,-1) - triu(b,1), where b = abs(a).
        comp(a, 1) is a with each diagonal element replaced by its
        absolute value, and each off-diagonal element replaced by minus
        the absolute value of the largest element in absolute value in
        its row.  however, if a is triangular comp(a, 1) is too.
        comp(a, 0) is the same as comp(a).
        comp(a) is often denoted by m(a) in the literature.

        Reference (e.g.):
        N.J. Higham, A survey of condition number estimation for
        triangular matrices, SIAM Review, 29 (1987), pp. 575-596.
    """
    m, n = a.shape
    p = min(m, n)

    if k == 0:

        # This code uses less temporary storage than the `high level'
        # definition above. (well, maybe... not clear that this is so
        # in numpy as opposed to m*lab)
        c = -abs(a)
        for j in range(p):
            c[j, j] = np.abs(a[j, j])

    elif k == 1:

        c = a.T
        for j in range(p):
            c[k, k] = 0

        mx = np.empty(p)
        for j in range(p):
            mx[j] = max(abs(c[j, :]))

        c = -np.outer(mx * np.ones(n))
        for j in range(p):
            c[j, j] = abs(a[j, j])

        if (a == np.tril(a)).all():
            c = np.tril(c)
        if (a == np.triu(a)).all():
            c = np.triu(c)

    return c
