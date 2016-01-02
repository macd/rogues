import numpy as np


def gearm(n, i=None, j=None):
    """
    GEARM   Gear matrix.
        A = GEARM(N,I,J) is the N-by-N matrix with ones on the sub- and
        super-diagonals, SIGN(I) in the (1,ABS(I)) position, SIGN(J)
        in the (N,N+1-ABS(J)) position, and zeros everywhere else.
        Defaults: i = n - 1, j = -(n - 1).  Note we use the zero based
        indexing for i and j.
        All eigenvalues are of the form 2*COS(a) and the eigenvectors
        are of the form [SIN(w+a), SIN(w+2a), ..., SIN(w+Na)].
        The values of a and w are given in the reference below.
        A can have double and triple eigenvalues and can be defective.
        GEARM(N) is singular.

        (GEAR is a Simulink function, hence GEARM for Gear matrix.)
        Reference:
        C.W. Gear, A simple set of test matrices for eigenvalue programs,
        Math. Comp., 23 (1969), pp. 119-125.
    """
    if i is None:
        i = n - 1
        j = -(n - 1)

    if not(abs(i) < n and abs(j) < n):
        raise ValueError('Invalid i and j parameters')

    a = np.diag(np.ones(n - 1), -1) + np.diag(np.ones(n - 1), 1)
    a[0, np.abs(i)] = np.sign(i)
    a[n - 1, n - 1 - np.abs(j)] = np.sign(j)

    return a
