import numpy as np


class Higham(Exception):
    pass


def wilkinson(n):
    """
    wilkinson array of size n where n must be odd.
    This is what some others call a Wilkinson array for arbitrary n. Note
    that Higham only uses this definition for n = 21
    """
    if not n % 2:
        raise Higham("n must be odd")

    m = int(n / 2)
    y = np.diag(np.abs(np.arange(-m, m + 1)))
    x1 = np.diag(np.ones(2 * m), 1)
    x2 = np.diag(np.ones(2 * m), -1)
    return y + x1 + x2
