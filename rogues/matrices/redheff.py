import numpy as np


def redheff(n):
    """
    redheff    a (0,1) matrix of redheffer associated with the
               riemann hypothesis.
           a = redheff(n) is an n-by-n matrix of 0s and 1s defined by
               a[i,j] = 1 if j = 0 or if (i+1) divides (j+1),
               a[i,j] = 0 otherwise.
           It has n - floor(log2(n)) - 1 eigenvalues equal to 1,
           a real eigenvalue (the spectral radius) approximately sqrt(n),
           a negative eigenvalue approximately -sqrt(n),
           and the remaining eigenvalues are provably ``small''.
           barrett and jarvis (1992) conjecture that
             ``the small eigenvalues all lie inside the unit circle
               abs(z) = 1'',
           and a proof of this conjecture, together with a proof that some
           eigenvalue tends to zero as n tends to infinity, would yield
           a new proof of the prime number theorem.
           The Riemann hypothesis is true if and only if
           det(a) = o( n^(1/2+epsilon) ) for every epsilon > 0
                                             (`!' denotes factorial).
           see also riemann().

           Reference:
           W.W. Barrett and T.J. Jarvis,
           Spectral Properties of a Matrix of Redheffer,
           Linear Algebra and Appl., 162 (1992), pp. 673-683.
    """
    i = np.outer(np.arange(1, n + 1), np.ones(n))
    a = np.where(np.remainder(i.T, i) == 0, 1, 0)
    a[:, 0] = np.ones(n)

    return a
