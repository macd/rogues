import numpy as np
from rogues.utils import cpltaxes


class Higham(Exception):
    pass


def ps(a, m=None, tol=1e-3, rl=0, marksize=0):
    """
    PS     Dot plot of a pseudospectrum.
       ps(a, m, tol, rl) plots an approximation to a pseudospectrum
       of the square matrix A, using M random perturbations of size TOL.
       M defaults to a SIZE(A)-dependent value and TOL to 1E-3.
       RL defines the type of perturbation:
         RL =  0 (default): absolute complex perturbations of 2-norm TOL.
         RL =  1:           absolute real perturbations of 2-norm TOL.
         RL = -1:           componentwise real perturbations of size TOL.
       The eigenvalues of A are plotted as crosses `x'.
       PS(A, M, TOL, RL, MARKSIZE) uses the specified marker size instead
       of a size that depends on the figure size, the matrix order, and M.
       If MARKSIZE < 0, the plot is suppressed and the plot data is returned
       as an output argument.
       PS(A, 0) plots just the eigenvalues of A.

       For a given TOL, the pseudospectrum of A is the set of
       pseudo-eigenvalues of A, that is, the set
       { e : e is an eigenvalue of A+E, for some E with NORM(E) <= TOL }.

       References:
       L. N. Trefethen, Computation of pseudospectra, Acta Numerica,
          8:247-295, 1999.
       L. N. Trefethen, Spectra and pseudospectra, in The Graduate
          Student's Guide to Numerical Analysis '98, M. Ainsworth,
          J. Levesley, and M. Marletta, eds., Springer-Verlag, Berlin,
          1999, pp. 217-250.
    """

    try:
        plt = __import__('pylab')
    except ImportError:
        raise Higham('Error importing pylab')

    if np.diff(a.shape)[0] != 0:
        raise Higham('Matrix must be square.')

    n = max(a.shape)

    if m is None:
        m = int(5 * max(1, np.around(25 * np.exp(-0.047 * n))))

    if m == 0:
        e, v = np.linalg.eig(a)
        ax = cpltaxes(e, plt)
        plt.plot(e.real, e.imag, 'rx')
        plt.axis(ax)
        plt.axis('equal')
        plt.show()
        return

    # If we don't create x with dtype=np.complex128, then
    # it defaults to float and the imaginary part of the eigenvalues
    # disappear (or we get an error, which is better) when we later
    # assign into x.
    x = np.empty(m * n, dtype=np.complex128)

    for j in range(m):
        # Componentwise.
        if rl == -1:
            # Uniform random numbers on [-1, 1].
            da = -np.ones(n) + 2 * np.random.rand(n, n)
            da = tol * (a * da)
        else:
            if rl == 0:
                # Complex absolute.
                da = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            else:
                # Real absolute.
                da = np.random.randn(n, n)

            da = (tol / np.linalg.norm(da)) * da

        x[j * n:(j + 1) * n], v = np.linalg.eig(a + da)

    if marksize >= 0:
        ax = cpltaxes(x, plt)
        h = plt.plot(x.real, x.imag, '.')
        plt.axis(ax)
        plt.axis('equal')

        if marksize == 0:
            # Apparently the 'units' property is not supported in
            # matplotlib and the postion property returns values
            # that have been normalized between 0 and 1 and are
            # a Bbox instance (a [2,2] array not a [4] array).  Not
            # clear how to proceed in picking the marker size for
            # the pseudo eigenvalues so we punt an put in a default
            marksize = 5

        e, v = np.linalg.eig(a)
        plt.plot(e.real, e.imag, 'rx')
        plt.setp(h, 'markersize', marksize)
        plt.show()

    else:
        y = x
        return y

    return None
