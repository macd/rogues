import numpy as np
import scipy.sparse as sparse


class Higham(Exception):
    pass


def tridiag(n, x=None, y=None, z=None):
    """
    tridiag  tridiagonal matrix (sparse).
         tridiag(x, y, z) is the sparse tridiagonal matrix with
         subdiagonal x, diagonal y, and superdiagonal z.
         x and z must be vectors of dimension one less than y.
         Alternatively tridiag(n, c, d, e), where c, d, and e are all
         scalars, yields the toeplitz tridiagonal matrix of order n
         with subdiagonal elements c, diagonal elements d, and superdiagonal
         elements e.   This matrix has eigenvalues (todd 1977)
                  d + 2*sqrt(c*e)*cos(k*pi/(n+1)), k=1:n.
         tridiag(n) is the same as tridiag(n,-1,2,-1), which is
         a symmetric positive definite m-matrix (the negative of the
         second difference matrix).

         References:
         J. Todd, Basic Numerical Mathematics, Vol. 2: Numerical Algebra,
           Birkhauser, Basel, and Academic Press, New York, 1977, p. 155.
         D.E. Rutherford, Some continuant determinants arising in physics and
           chemistry---II, Proc. Royal Soc. Edin., 63, A (1952), pp. 232-241.
    """
    try:
        # First see if they are arrays
        nx, = n.shape
        ny, = x.shape
        nz, = y.shape
        if (ny - nx - 1) != 0 or (ny - nz - 1) != 0:
            raise Higham('Dimensions of vector arguments are incorrect.')
        # Now swap to match above
        z = y
        y = x
        x = n

    except AttributeError:
        # They are not arrays
        if n < 2:
            raise Higham("n must be 2 or greater")

        if x is None and y is None and z is None:
            x = -1
            y = 2
            z = -1

        x = x * np.ones(n - 1)
        z = z * np.ones(n - 1)
        y = y * np.ones(n)

    except ValueError:
        raise Higham("x, y, z must be all scalars or 1-D vectors")

    # t = diag(x, -1) + diag(y) + diag(z, 1);  % For non-sparse matrix.
    n = np.max(np.size(y))
    za = np.zeros(1)

    # Use the (*)stack functions instead of the r_[] notation in
    # an attempt to be more readable. (Doesn't look like it helped much)
    t = sparse.spdiags(np.vstack((np.hstack((x, za)), y,
                                  np.hstack((za, z)))),
                       np.array([-1, 0, 1]), n, n)

    return t
