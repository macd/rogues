import numpy as np


class Higham(Exception):
    pass


def repmat(a, repeat):
    """
    repmat(a, repeat)
    Simple implementation of m*lab's repmat function.
    repeat is assumed to be a 2-tuple.
    """
    if len(repeat) != 2:
        raise Higham("repeat must be a two-tuple")

    m, n = repeat
    b = a
    for i in range(1, n):
        b = np.hstack((b, a))

    c = b
    for i in range(1, m):
        c = np.vstack((c, b))

    return c
