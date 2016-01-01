import numpy as np


def pow2(x, y=None):
    """
    Raise 2 to the power of x[i] for the vector x.  If two vectors
    are supplied, return    x[i] * (2 ** y[i])
    Note that no error checking is done in this example.
    """
    if y is None:
        z = (2. * np.ones(len(x))) ** x
    else:
        z = x * ((2. * np.ones(len(y))) ** y)

    return z
