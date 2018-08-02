import numpy as np


def cycol(mn, k=None):
    """
    cycol   matrix whose columns repeat cyclically.
        a = cycol(mn, k) (mn is a 2-tuple) is an m-by-n matrix of
        the form a = b(1:m,1:n) where b = [c c c...] and c = randn(m, k).
        Thus a's columns repeat cyclically, and a has rank at most k.
        k need not divide n. k defaults to round(n/4).
        cycol(n, k), where n is a scalar, is the same as cycol((n n), k).

        This type of matrix can lead to underflow problems for Gaussian
        elimination: see NA Digest Volume 89, Issue 3 (January 22, 1989).
    """
    try:
        m, n = mn
    except (TypeError, AttributeError):
        m = mn
        n = mn

    if k is None:
        k = max(n // 4, 1)
    else:
        if k > n:
            raise ValueError("k cannot be greater than the "
                             "max matrix dimension")

    a = np.random.randn(m, k)

    for i in range(1, int(np.ceil(n / k))):
        a = np.hstack((a, a[:, 0:k]))

    # Truncate matrix down to desired size if we concat'ed too much
    a = a[:, 0:n]

    return a
