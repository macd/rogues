import numpy as np


def chebspec(n, k=0):
    """"
    CHEBSPEC  Chebyshev spectral differentiation matrix.
          c = chebspec(n, k) is a Chebyshev spectral differentiation
          matrix of order n.  k = 0 (the default) or 1.
          For k = 0 (`no boundary conditions'), c is nilpotent, with
              c**n = 0 and it has the null vector ones(n,1).
              c is similar to a Jordan block of size n with eigenvalue zero.
              Default is k = 0
          For k = 1, c is nonsingular and well-conditioned, and its eigenvalues
              have negative real parts.
          For both k, the computed eigenvector matrix x from eig is
              ill-conditioned (mesh(real(x)) is interesting).

          References:
          C. Canuto, M.Y. Hussaini, A. Quarteroni and T.A. Zang, Spectral
             Methods in Fluid Dynamics, Springer-Verlag, Berlin, 1988; p. 69.
          L.N. Trefethen and M.R. Trummer, An instability phenomenon in
             spectral methods, SIAM J. Numer. Anal., 24 (1987), pp. 1008-1023.
          D. Funaro, Computing the inverse of the Chebyshev collocation
             derivative, SIAM J. Sci. Stat. Comput., 9 (1988), pp. 1050-1057.
    """

    # k = 1 case obtained from k = 0 case with one bigger n.
    if k == 1:
        n += 1

    n = n - 1
    c = np.zeros((n + 1, n + 1))

    one = np.ones(n + 1)
    x = np.cos(np.arange(0, n + 1) * (np.pi / float(n)))
    d = np.ones(n + 1)
    d[0] = 2
    d[n] = 2

    # np.eye(n + 1) in next expression avoids div by zero.
    c = np.outer(d, (one / d)) / (np.outer(x, one) -
                                  np.outer(one, x) + np.eye(n + 1))

    #  Now fix diagonal and signs.
    c[0, 0] = (2 * n ** 2 + 1) / 6.0
    for i in range(1, n + 1):
        if ((i + 1) % 2) == 0:
            c[:, i] = -c[:, i]
            c[i, :] = -c[i, :]

        if i < n:
            c[i, i] = -x[i] / (2 * (1 - x[i] ** 2))
        else:
            c[n, n] = -c[0, 0]

    if k == 1:
        c = c[1::, 1::]

    return c
