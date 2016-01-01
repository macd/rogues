import numpy as np


def vecperm(m, n=None):
    """
    VECPERM    Vec-permutation matrix.
           VECPERM(M, N) is the vec-permutation matrix, an MN-by-MN
           permutation matrix P with the property that if A is M-by-N then
           vec(A) = P*vec(A').
           If N is omitted, it defaults to M.

           P is formed by taking every n'th row from EYE(M*N), starting with
           the first and working down - see p. 277 of the reference.

           Reference:
           H. V. Henderson and S. R. Searle The vec-permutation matrix,
           the vec operator and Kronecker products: A review Linear and
           Multilinear Algebra, 9 (1981), pp. 271-288.
    """
    if n is None:
        n = m

    p = np.zeros((m * n, m * n))
    e = np.eye(m * n)

    k = 0
    for i in range(n):
        p[k:k + n, :] = e[i:m * n:n, :]
        k += n

    return p
