import numpy as np
import numpy.linalg as nl
import pylab as plt
from rogues.utils import rq, cpltaxes


class Higham(Exception):
    pass


def fv(b, nk=1, thmax=16, do_plot=True):
    """
    FV     Field of values (or numerical range).
       fv(b, nk, thmax, do_plot) evaluates and plots the field of values of the
       NK largest leading principal submatrices of b, using THMAX
       equally spaced angles in the complex plane.
       The defaults are NK = 1 and THMAX = 16.
       (For a `publication quality' picture, set THMAX higher, say 32.)
       The eigenvalues of b are displayed as `x'.
       Alternative usage: f, e = fv(a, nk, thmax, False) suppresses the
       plot and returns the field of values plot data in F, with A's
       eigenvalues in E.   Note that norm(f, inf) approximates the
       numerical radius,
                 max {abs(z): z is in the field of values of A}.

       Theory:
       Field of values fv(a) = set of all Rayleigh quotients. fv(a) is a
       convex set containing the eigenvalues of A.  When A is normal fv(a) is
       the convex hull of the eigenvalues of A (but not vice versa).
               z = x.T * a * x / np.dot(x,x),
               z.T = x.T * a.T * x / np.dot(x,x)
               => z.real = x.T * h * x / np.dot(x,x),   h = (a + a.T)/2
       so      min(eig(h)) <= z.real <= max(eig(h)),
       with equality for x = corresponding eigenvectors of h.  For these x,
       rq(a,x) is on the boundary of fv(a).

       Based on an original routine by A. Ruhe.

       References:
       R. A. Horn and C. R. Johnson, Topics in Matrix Analysis, Cambridge
            University Press, 1991; sec. 1.5.
       A. S. Householder, The Theory of Matrices in Numerical Analysis,
            Blaisdell, New York, 1964; sec. 3.3.
       C. R. Johnson, Numerical determination of the field of values of a
            general complex matrix, SIAM J. Numer. Anal., 15 (1978),
            pp. 595-602.

       Note added in porting... changed the last parameter's name.  Now it
       will suppress ploting when set to False.
    """

    thmax = thmax - 1     # Because code below uses thmax + 1 angles.

    if len(b.shape) != 2:
        raise Higham('Matrix must be two dimensional.')
    else:
        n, p = b.shape
        if n != p:
            raise Higham('Matrix must be square.')

    f = None
    z = np.zeros(2*thmax + 1, dtype=np.complex128)
    e, v = nl.eig(b)

    # Filter out cases where B is Hermitian or skew-Hermitian, for efficiency.
    if (b == b.T).all():
        f = np.hstack((np.min(e), np.max(e)))

    elif (b == -b.T).all():
        e = e.imag
        f = np.hstack((np.min(e), np.max(e)))
        e = 1j * e
        f = 1j * f

    else:
        for m in range(nk):
            ns = n - m
            a = b[0:ns, 0:ns]

            for i in range(thmax):
                th = i / float(thmax) * np.pi
                # Rotate A through angle th.
                ath = np.exp(1j * th) * a
                # Hermitian part of rotated A.
                h = 0.5 * (ath + ath.T)
                d, x = nl.eig(h)
                k = np.argsort(d.real)
                # rq's of a corr. to eigenvalues of h
                z[i] = rq(a, x[:, k[0]])
                # with smallest/largest real part.
                z[i + thmax] = rq(a, x[:, k[ns-1]])

            if f is None:
                f = z
            else:
                f = np.vstack((f, z))

        # Next line ensures boundary is `joined up' (needed for orthogonal
        # matrices).
        if len(f.shape) > 1:
            f = np.vstack((f, f[0, :]))
        else:
            f = np.vstack((f, f))

    if thmax == 0:
        f = e

    if do_plot:
        ax = cpltaxes(f, plt)
        # Plot the field of values
        plt.plot(f.real, f.imag, 'o')
        plt.axis(ax)
        plt.axis('equal')
        # Plot the eigenvalues too.
        plt.plot(e.real, e.imag, 'x')

    plt.show()
    return f, e
