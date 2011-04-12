import numpy as np


class Higham(Exception):
    pass


def mgs(a):
    """
    MGS     Modified Gram-Schmidt QR factorization.
        q, r = mgs(a) uses the modified Gram-Schmidt method to compute the
        factorization a = q*e for m-by-n a of full rank,
        where q is m-by-n with orthonormal columns and R is n-by-n.
    """
    try:
        m, n = a.shape
    except AttributeError:
        raise Higham("Input array must be two dimensional")

    q = np.zeros((m, n))
    r = np.zeros((n, n))

    for k in range(n):
        r[k, k] = np.linalg.norm(a[:, k])
        q[:, k] = a[:, k] / r[k, k]
        r[k, k + 1:n] = np.dot(q[:, k], a[:, k + 1:n])
        a[:, k + 1:n] = a[:, k + 1:n] - np.outer(q[:, k], r[k, k + 1:n])

    return q, r
