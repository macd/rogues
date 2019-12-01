import numpy as np
import scipy as sp
from rogues.utils import cpltaxes
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Higham(Exception):
    pass


def pscont(a, k=0, npts=None, ax=None, levels=None):
    """
    NOTE: Porting to Python/Numpy/Matplotlib has been less than
          ideal.  It currently only 'sorta, kinda works'

    pscont   Contours and colour pictures of pseudospectra.
         pscont(a, k, npts, ax, levels) plots LOG10(1/NORM(R(z))),
         where R(z) = INV(z*I-A) is the resolvent of the square matrix A,
         over an npts-by-npts grid.
         npts defaults to a SIZE(A)-dependent value.
         The limits are ax[0] and ax[1] on the x-axis and
                        ax[2] and ax[3] on the y-axis.
         If ax is omitted, suitable limits are guessed based on the
         eigenvalues of A.
         The eigenvalues of A are plotted as crosses `x'.
         k determines the type of plot:
             k = 0 (default) PCOLOR and CONTOUR
             k = 1           PCOLOR only
             k = 2           SURFC (SURF and CONTOUR)
             k = 3           SURF only
             k = 4           CONTOUR only
         The contours levels are specified by the vector LEVELS, which
         defaults to -10:-1 (recall we are plotting log10 of the data).
         Thus, by default, the contour lines trace out the boundaries of
         the epsilon pseudospectra for epsilon = 1e-10, ..., 1e-1.
         [X, Y, Z, NPTS] = PSCONT(A, ...) returns the plot data X, Y, Z
         and the value of NPTS used.

         After calling this function you may want to change the
         color map (e.g., type COLORMAP HOT - see HELP COLOR) and the
         shading (e.g., type SHADING INTERP - see HELP INTERP).
         For an explanation of the term `pseudospectra', and references,
         see PS.M.
         When A is real and the grid is symmetric about the x-axis, this
         routine exploits symmetry to halve the computational work.

         Colour pseduospectral pictures of this type are referred to as
         `spectral portraits' by Godunov, Kostin, and colleagues.
         References: see PS.
    """

    if np.diff(a.shape)[0] != 0:
        raise Higham('Matrix must be square.')

    n = np.max(a.shape)
    is_a_real = not (a.imag).any()

    if levels is None:
        levels = np.arange(-10, 0)

    e, v = np.linalg.eig(a)

    if ax is None:
        ax = cpltaxes(e, plt)
        if is_a_real:
            ax[2] = -ax[3]

    if npts is None:
        npts = int(3*np.round(min(max(5, np.sqrt(20**2 * 10**3 / n**3)), 30)))

    nptsx = npts
    nptsy = npts
    # ysymmetry = is_a_real and (ax[2] == -ax[3])
    ysymmetry = False  # Hack... to be fixed

    x = np.linspace(ax[0], ax[1], npts)
    y = np.linspace(ax[2], ax[3], npts)
    if ysymmetry:                        # Exploit symmetry about x-axis.
        nptsy = int(np.ceil(npts/2))
        y1 = y
        y = y[0:nptsy]

    xx, yy = np.meshgrid(x, y)
    z = xx + 1j * yy
    eye_n = np.eye(n)
    smin = np.zeros((nptsy, nptsx))

    for j in range(nptsx):
        for i in range(nptsy):
            u, s, v = sp.linalg.svd(z[i, j] * eye_n - a)
            smin[i, j] = np.min(s)

    z = np.log10(smin + np.finfo(float).eps)
    if ysymmetry:
        z = np.vstack((z, z[nptsy - np.remainder(npts, 2)::-1, :]))
        y = y1

    if k == 0 or k == 1:
        plt.pcolor(x, y, z)
    elif k == 2 or k == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1,
                               cmap=cm.coolwarm, linewidth=.2,
                               antialiased=True)

    if k == 0:
        plt.contour(x, y, z, levels)
    elif k == 4:
        plt.contour(x, y, z, levels)

    if k != 2 and k != 3:
        if k == 0 or k == 1:
            s = 'w'   # White
        else:
            s = 'k'   # Black
        plt.plot(e.real, e.imag, ''.join((s, 'x')))

    plt.axis('equal')
    plt.show()
    return x, y, z
