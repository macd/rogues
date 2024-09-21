import numpy as np
import numpy.linalg as nl
from rogues.matrices import qmult
from rogues.utils import bandred


class Higham(Exception):
    pass


def randsvd(n, kappa=None, mode=3, kl=None, ku=None):
    """
    RANDSVD  Random matrix with pre-assigned singular values.
      randsvd(n, kappa, mode, kl, ku) is a (banded) random matrix of order n
      with cond(a) = kappa and singular values from the distribution mode.
      n may be a 2-tuple, in which case the matrix is n[0]-by-n[1].
      Available types:
             mode = 1:   one large singular value,
             mode = 2:   one small singular value,
             mode = 3:   geometrically distributed singular values,
             mode = 4:   arithmetically distributed singular values,
             mode = 5:   random singular values with unif. dist. logarithm.
      If omitted, mode defaults to 3, and kappa defaults to sqrt(1/eps).
      If mode < 0 then the effect is as for abs(mode) except that in the
      original matrix of singular values the order of the diagonal entries
      is reversed: small to large instead of large to small.
      KL and KU are the lower and upper bandwidths respectively; if they
      are omitted a full matrix is produced.
      If only KL is present, KU defaults to KL.
      Special case: if KAPPA < 0 then a random full symmetric positive
                    definite matrix is produced with cond(a) = -kappa and
                    eigenvalues distributed according to mode.
                    kl and ku, if present, are ignored.

      Reference:
      N.J. Higham, Accuracy and Stability of Numerical Algorithms,
         Society for Industrial and Applied Mathematics, Philadelphia, PA,
         USA, 2002; sec. 28.3.

      This routine is similar to the more comprehensive Fortran routine xLATMS
      in the following reference:
      J.W. Demmel and A. McKenney, A test matrix generation suite,
      LAPACK Working Note #9, Courant Institute of Mathematical Sciences,
      New York, 1989.
    """
    # Parameter n specifies dimension: m-by-n.
    try:
        m, n = n
        p = min(m, n)
    except TypeError:
        m = n
        p = n

    if kappa is None:
        kappa = np.sqrt(1 / np.finfo(float).eps)

    if kl is None:
        kl = n - 1          # Full matrix.

    if ku is None:
        ku = kl             # Same upper and lower bandwidths.

    if np.abs(kappa) < 1:
        raise Higham('Condition number must be at least 1!')

    is_posdef = False
    if kappa < 0:
        is_posdef = True
        kappa = -kappa

    if p == 1:              # Handle case where a is a vector, not a matrix
        a = np.random.randn(max(m, n))
        a = a / nl.norm(a)
        return a

    j = np.abs(mode)

    # Set up vector sigma of singular values.
    if j == 3:
        factor = kappa ** (-1 / (p - 1))
        sigma = factor ** np.arange(0, p)

    elif j == 4:
        sigma = np.ones(p) - np.arange(p) / (p - 1) * (1 - 1 / kappa)

    elif j == 5:
        # In this case cond(a) <= kappa
        sigma = np.exp(-np.random.rand(size=p) * np.log(kappa))

    elif j == 2:
        sigma = np.ones(p)
        sigma[p - 1] = 1 / kappa

    elif j == 1:
        sigma = np.ones(p) / kappa
        sigma[0] = 1

    # Convert to diagonal matrix of singular values.
    if mode < 0:
        sigma = sigma[::-1]

    sigma = np.diag(sigma)

    if is_posdef:               # Handle special case.
        q = qmult(p)
        a = q.T @ sigma @ q
        a = (a + a.T) / 2.      # Ensure matrix is symmetric.
        return a

    # Expand matrix, if necessary
    if m > n:
        sigma = np.vstack((sigma, np.zeros((m - n, n))))
    elif m < n:
        sigma = np.hstack((sigma, np.zeros((m, n - m))))

    if kl == 0 and ku == 0:  # Diagonal matrix requested - nothing more to do.
        a = sigma
        return a

    # A = U*sigma*V, where U, V are random orthogonal matrices from the
    # Haar distribution.
    a = qmult(sigma.T)
    a = qmult(a.T)

    if kl < n - 1 or ku < n - 1:   # Bandwidth reduction.
        a = bandred(a, kl, ku)

    return a
