import numpy as np
from rogues.matrices.jordbloc import jordbloc


def forsythe(n, alpha=None, lambduh=0):
    """
    forsythe  forsythe matrix - a perturbed jordan block.
          forsythe(n, alpha, lambda) is the n-by-n matrix equal to
          jordbloc(n, lambda) except it has an alpha in the [n - 1, 0] position
          It has the characteristic polynomial
                  det(a-t*eye) = (lambda-t)^n - (-1)^n alpha.
          alpha defaults to sqrt(eps) and lambduh to 0.
    """
    if alpha is None:
        alpha = np.sqrt(np.finfo(float).eps)

    a = jordbloc(n, lambduh)
    a[n - 1, 0] = alpha

    return a
