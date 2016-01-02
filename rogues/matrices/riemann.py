import numpy as np


def riemann(n):
    """
    RIEMANN    A matrix associated with the Riemann hypothesis.
           A = RIEMANN(N) is an N-by-N matrix for which the
           Riemann hypothesis is true if and only if
           DET(A) = O( N! N^(-1/2+epsilon) ) for every epsilon > 0
                                             (`!' denotes factorial).
           A = B(2:N+1, 2:N+1), where
           B(i,j) = i-1 if i divides j and -1 otherwise.
           Properties include, with M = N+1:
              Each eigenvalue E(i) satisfies ABS(E(i)) <= M - 1/M.
              i <= E(i) <= i+1 with at most M-SQRT(M) exceptions.
              All integers in the interval (M/3, M/2] are eigenvalues.

           See also REDHEFF.

           Reference:
           F. Roesler, Riemann's hypothesis as an eigenvalue problem,
           Linear Algebra and Appl., 81 (1986), pp. 153-198.
    """
    n = n + 1
    i = np.outer(np.arange(2, n + 1), np.ones(n - 1))
    j = i.T
    a = np.where(np.remainder(j, i) == 0, i, -1)

    return a
