import numpy as np


def cgs(a):
    """
    CGS     Classical Gram-Schmidt QR factorization.
    [Q, R] = cgs(A) uses the classical Gram-Schmidt method to compute the
    factorization A = Q*R for m-by-n A of full rank,
    where Q is m-by-n with orthonormal columns and R is n-by-n.

    NOTE: This was ported just for an example of the Gram-Schmidt
    orthogonalization. If you really want the QR factorization, you
    should probably use numpy.linalg.qr
    """
    m, n = a.shape
    q = np.zeros((m, n))
    r = np.zeros((n, n))

    # Gram-Schmidt expressed in matrix-vector form.
    # Treat j = 0 as special case to avoid if tests inside loop.
    r[0, 0] = np.linalg.norm(a[:, 0])
    q[:, 0] = a[:, 0] / r[0, 0]
    for j in range(1, n):
        r[0:j - 1, j] = np.dot(q[:, 0:j - 1].T, a[:, j])
        temp = a[:, j] - np.dot(q[:, 0:j - 1], r[0:j - 1, j])
        r[j, j] = np.linalg.norm(temp)
        q[:, j] = temp / r[j, j]

    return q, r
