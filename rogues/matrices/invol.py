from rogues.matrices import hilb


def invol(n):
    """
    invol(n)   an involutory matrix of order n.
        a = invol(n) is an n-by-n involutory (a*a = eye(n)) and
        ill-conditioned matrix.
        It is a diagonally scaled version of hilb(n).
        nb: b = (eye(n)-a)/2 and b = (eye(n)+a)/2 are idempotent (b*b = b).

        Reference:
        A.S. Householder and J.A. Carpenter, The singular values
        of involutory and of idempotent matrices, Numer. Math. 5 (1963),
        pp. 234-237.
    """

    a = hilb(n)

    d = -n
    a[:, 0] = d * a[:, 0]

    for i in range(n - 1):
        d = -(n + i + 1) * (n - i - 1) * d / float((i + 1) * (i + 1))
        a[i + 1, :] = d * a[i + 1, :]

    return a
