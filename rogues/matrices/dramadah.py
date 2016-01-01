import numpy as np
import rogues


def dramadah(n, k=1):
    """
    dramadah  a (0,1) matrix whose inverse has large integer entries.
          An anti-hadamard matrix a is a matrix with elements 0 or 1 for
          which mu(a) := norm(inv(a),'fro') is maximal.
          a = dramadah(n, k) is an n-by-n (0,1) matrix for which mu(a) is
          relatively large, although not necessarily maximal.
          Available types (the default is k = 1):
          k = 1: a is toeplitz, with abs(det(a)) = 1, and mu(a) > c(1.75)^n,
                 where c is a constant.
          k = 2: a is upper triangular and toeplitz.
          the inverses of both types have integer entries.

          Another interesting (0,1) matrix:
          k = 3: A has maximal determinant among (0,1) lower Hessenberg
          matrices: det(A) = the n'th Fibonacci number.  A is Toeplitz.
          The eigenvalues have an interesting distribution in the complex
          plane.

          References:
          R.L. Graham and N.J.A. Sloane, Anti-Hadamard matrices,
             Linear Algebra and Appl., 62 (1984), pp. 113-137.
          L. Ching, The maximum determinant of an nxn lower Hessenberg
             (0,1) matrix, Linear Algebra and Appl., 183 (1993), pp. 147-153.
    """

    if k == 1:
        # Toeplitz
        c = np.ones(n)
        for i in range(1, n, 4):
            m = min(1, n - i)
            c[i:i + m + 1] = 0

        r = np.zeros(n)
        r[0:4] = np.array([1, 1, 0, 1])
        if n < 4:
            r = r[0:n]

        a = rogues.toeplitz(c, r)

    elif k == 2:
        # Upper triangular and Toeplitz
        c = np.zeros(n)
        c[0] = 1
        r = np.ones(n)
        r[2::2] = 0

        a = rogues.toeplitz(c, r)

    elif k == 3:
        # Lower Hessenberg.
        c = np.ones(n)
        c[1::2] = 0
        a = rogues.toeplitz(c, np.hstack((np.array([1, 1]), np.zeros(n - 2))))

    return a
