import numpy as np
import scipy.sparse as sparse


class Higham(Exception):
    pass


def wathen(nx, ny, k=0):
    """
    WATHEN  Wathen matrix - a finite element matrix (sparse, random entries).
        a = wathen(nx, ny) is a sparse random n-by-n finite element matrix
        where n = 3*nx*ny + 2*nx + 2*ny + 1.
        A is precisely the `consistent mass matrix' for a regular NX-by-NY
        grid of 8-node (serendipity) elements in 2 space dimensions.
        A is symmetric positive definite for any (positive) values of
        the `density', rho(nx,ny), which is chosen randomly in this routine.
        in particular, if d = diag(diag(a)), then
              0.25 <= eig(inv(d)*a) <= 4.5
        for any positive integers nx and ny and any densities rho(nx,ny).
        this diagonally scaled matrix is returned by wathen(nx,ny,1).
        For k = 1 wathen returns the diagonally scaled matrix.

        Reference:
        A.J. Wathen, Realistic eigenvalue bounds for the Galerkin
        mass matrix, IMA J. Numer. Anal., 7 (1987), pp. 449-457.
    """
    e1 = np.array([[6, -6, 2, -8],
                   [-6, 32, -6, 20],
                   [2, -6, 6, -6],
                   [-8, 20, -6, 32]])

    e2 = np.array([[3, -8, 2, -6],
                   [-8, 16, -8, 20],
                   [2, -8, 3, -8],
                   [-6, 20, -8, 16]])

    ea = np.hstack((e1, e2))
    eb = np.hstack((e2.T, e1))
    e = np.vstack((ea, eb)) / 45.
    n = 3 * nx * ny + 2 * nx + 2 * ny + 1
    a = sparse.lil_matrix((n, n))

    rho = np.random.randint(1, 100, (nx, ny))
    nn = np.zeros(8)

    for j in range(ny):
        for i in range(nx):
            nn[0] = 3 * j * nx + 3 * nx + 2 * i + 2 * j + 4
            nn[1] = nn[0] - 1
            nn[2] = nn[1] - 1
            nn[3] = (3 * j + 2) * nx + 2 * j + i + 1
            nn[4] = 3 * j * nx + 2 * i + 2 * j
            nn[5] = nn[4] + 1
            nn[6] = nn[5] + 1
            nn[7] = nn[3] + 1
            nn = nn.astype(int)
            
            em = e * rho[i, j]

            for krow in range(8):
                for kcol in range(8):
                    a[nn[krow], nn[kcol]] = a[nn[krow], nn[kcol]] + \
                                            em[krow, kcol]

    if k == 1:
        raise Higham("k = 1 option not supported... ignoring")

    return a
