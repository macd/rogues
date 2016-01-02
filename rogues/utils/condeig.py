import numpy as np
import scipy.linalg as sl


class Higham(Exception):
    pass


def condeig(a):
    """
    v, lambda, c = condeig(a) Computes condition numbers for the
    eigenvalues of a matrix. The condition numbers are the reciprocals
    of the cosines of the angles between the left and right eigenvectors.
    Inspired by Arno Onken's Octave code for condeig

    When checking against results obtained in Higham & Higham
    a = rogues.frank(6)
    lr, vr, c = matrixcomp.condeig(a)

    H & H get for lr  = [ 12.9736, 5.3832, 1.8355,  0.5448,  0.0771,  0.1858]
          and for  c  = [  1.3059, 1.3561, 2.0412, 15.3255, 43.5212, 56.6954]

    which they say that the small eigenvalues are slightly ill conditioned.
    With the this python/numpy condeig we get

    vr = [ 12.97360792, 5.38322318, 1.83552324, 0.54480378,   0.07707956,
            0.18576231]
     c = [  1.30589002, 1.35605093, 2.04115713, 15.32552609, 43.52124194,
           56.69535399]

    NOTE: we must use scipy.linalg.decomp.eig and _not_ np.linalg.eig
    """

    if len(a.shape) != 2:
        raise Higham("a must be a 2 dimensional array")
    else:
        m, n = a.shape
        if m != n or m < 2:
            raise Higham("a must be a square array with dimension of "
                         "2 or greater")

    # eigenvalues, left and right eigenvectors
    lamr, vl, vr = sl.eig(a, left=True, right=True)

    # Need to put the left eigenvectors into row form
    vl = vl.T

    # Normalize vectors
    for i in range(n):
        vl[i, :] = vl[i, :] / np.sqrt(abs(vl[i, :] ** 2).sum())

    # Condition numbers are reciprocal of the cosines (dot products) of the
    # left eignevectors with the right eigenvectors.   In a perfect world,
    # these numbers should all be one, but they are not.
    c = abs(1 / np.diag(np.dot(vl, vr)))

    return lamr, vr, c
