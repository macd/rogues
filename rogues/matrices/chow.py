import numpy as np
from rogues.utils import toeplitz


def chow(n, alpha=1, delta=0):
    """
    chow(n, alpha, delta):  chow matrix - a singular toeplitz
        lower hessenberg matrix.
        a = chow(n, alpha, delta) is a toeplitz lower hessenberg matrix
        a = h(alpha) + delta*eye, where h(i,j) = alpha^(i-j+1).
        h(alpha) has p = floor(n/2) zero eigenvalues, the rest being
        4*alpha*cos( k*pi/(n+2) )^2, k=1:n-p.
        defaults: alpha = 1, delta = 0.

        References:
        T.S. Chow, A class of Hessenberg matrices with known
           eigenvalues and inverses, SIAM Review, 11 (1969), pp. 391-395.
        G. Fairweather, On the eigenvalues and eigenvectors of a class of
           Hessenberg matrices, SIAM Review, 13 (1971), pp. 220-221.
    """
    a = toeplitz(alpha ** np.arange(1, n + 1),
                 np.hstack((alpha, 1, np.zeros(n - 2)))) + delta * np.eye(n)
    return a
