import numpy as np
import scipy.sparse as sparse


def pentoep(n, a=1, b=-10, c=0, d=10, e=1):
    """
    PENTOEP   Pentadiagonal Toeplitz matrix (sparse).
          p = pentoep(n, a, b, c, d, e) is the n-by-n pentadiagonal
          Toeplitz matrix with diagonals composed of the numbers
          a =: p[3,1], b =: p[2,1], c =: p[1,1], d =: p[1,2], e =: p[1,3].
          default: (a,b,c,d,e) = (1,-10,0,10,1) (a matrix of rutishauser).
                    This matrix has eigenvalues lying approximately on
                    the line segment 2*cos(2*t) + 20*i*sin(t).

          Interesting plots are
          ps(full(pentoep(32,0,1,0,0,1/4)))  - `triangle'
          ps(full(pentoep(32,0,1/2,0,0,1)))  - `propeller'
          ps(full(pentoep(32,0,1/2,1,1,1)))  - `fish'

          References:
          R.M. Beam and R.F. Warming, The asymptotic spectra of
             banded Toeplitz and quasi-Toeplitz matrices, SIAM J. Sci.
             Comput. 14 (4), 1993, pp. 971-1006.
          H. Rutishauser, On test matrices, Programmation en Mathematiques
             Numeriques, Editions Centre Nat. Recherche Sci., Paris, 165,
             1966, pp. 349-365.
    """
    o1 = np.ones(n)
    data = np.vstack((a * o1, b * o1, c * o1, d * o1, e * o1))
    diags = np.arange(-2, 3)
    p = sparse.spdiags(data, diags, n, n)

    return p.todense()
