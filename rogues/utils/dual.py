import numpy as np


class Higham(Exception):
    pass


def dual(x, p=None):
    """
    DUAL    Dual vector with respect to Holder p-norm.
        y = dual(x, p), where 1 <= p <= inf, is a vector of unit q-norm
        that is dual to X with respect to the p-norm, that is,
        norm(Y, q) = 1 where 1/p + 1/q = 1 and there is
        equality in the Holder inequality: X'*Y = norm(X, p)*norm(Y, q).
        Special case: DUAL(X), where X >= 1 is a scalar, returns Y such
                      that 1/X + 1/Y = 1.

        Called by PNORM.
    """

    if p is None:
        if len(x) == 1:
            y = 1 / (1 - 1 / x)
            return
        else:
            raise Higham('Second argument missing.')

    q = 1 / (1 - 1 / float(p))

    if np.linalg.norm(x, np.inf) == 0:
        y = x
        return

    if p == 1:
        # we want zero to be thought of as "positive"
        y = np.where(np.sign(x) == 0, 1, np.sign(x))

    elif p == np.inf:
        # y is a multiple of unit vector e_k.
        y = np.where(np.abs(x) == np.abs(x).max, np.sign(x), 0)

    else:
        # 1 < p < inf.  Dual is unique in this case.
        # This scaling helps to avoid under/over-flow.
        x = x / np.linalg.norm(x, np.inf)
        y = np.abs(x) ** (p - 1) * np.where(np.sign(x) == 0, 1, np.sign(x))
        y = y / np.linalg.norm(y, q)         # Normalize to unit q-norm.

    return y
