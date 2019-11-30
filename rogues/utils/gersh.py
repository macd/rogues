import numpy as np
import matplotlib.pylab as plt
from rogues.utils import cpltaxes


class Higham(Exception):
    pass


def gersh(A, plot=True):
    ''''
    gersh    Gershgorin disks.
             gersh(A) draws the Gershgorin disks for the square matrix A.
             The eigenvalues are plotted as crosses `x'.
             Alternative usage: [G, E] = gersh(a, False) suppresses the plot
             and returns the data in G, with A's eigenvalues in E.

             Try gersh(rg.lesp(n)) and gersh(rg.smoke(n))
    '''

    if not (A.ndim == 2 and np.diff(A.shape)[0] == 0):
        raise Higham("Matrix must be square.  It is: ", A.shape)

    n = len(A)
    m = 40
    G = np.zeros((m, n), dtype=np.complex128)

    d = np.diag(A)
    r = sum(np.abs(A - np.diag(d)).T).T
    e, _ = np.linalg.eig(A)

    radvec = np.exp(1j * np.linspace(0, 2 * np.pi, m))

    for j in range(n):
        G[:, j] = d[j] * np.ones(m) + r[j] * radvec

    if plot:
        ax = cpltaxes(G[:], plt)
        for j in range(n):
            plt.plot(G[:, j].real, G[:, j].imag, '-')      # Plot the disks.

        plt.plot(e.real, e.imag, 'x')    # Plot the eigenvalues too.
        plt.axis(ax)
        plt.axis('equal')
        plt.show()

    return G, e
