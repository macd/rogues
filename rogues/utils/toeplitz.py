import numpy as np


class Higham(Exception):
    pass


def toeplitz(a, b=None):
    """
    toeplitz(a) returns a toeplitz matrix given "a", the first row of the
    matrix.  This matrix is defined as:

        [ [   a[0], a[1], a[2], a[3], ...,   a[n] ],
          [   a[n], a[0], a[1], a[2], ..., a[n-1] ],
          [ a[n-1], a[n], a[0], a[1], ..., a[n-2] ],
          ...
          [   a[1], a[2], a[3], a[4], ...,   a[0] ]]

    Note that this array is more properly called a "circulant" because each
    row is a circular shift of the one above it. Also, note that the main
    diagonal is constant and given by a[0].  If the array "a" is complex,
    the elements rotated to be under the main diagonal are complex conjugated
    to yield a Hermetian matrix.

    If called as toeplitz(a, b) then create the unsymetric toeplitz matrix.
    Here "b" would be the first row and "a" would be the first column. (This
    is bass-ackwards to follow the m*lab convention.) The second row shifts
    the first row to the right by one, but instead circularly shifting the
    last element of the row back into the first element, we shift in the
    successive elements of a as the element of b are shifted out.

    See the wikipedia entry for more information and references (especially the
    following: http://ee.stanford.edu/~gray/toeplitz.pdf)
    """

    # Error checking...
    try:
        m, = a.shape
        if b is not None:
            n, = b.shape
    except (ValueError, AttributeError):
        raise Higham("Input arrays must be one dimensional")

    if b is not None:
        if m != n:
            raise Higham("Input arrays must be have the same dimension")
    else:
        # If only one array specified and it is complex, make sure that the
        # generated toeplitz array is Hermitian
        b = np.conj(a.copy())

    if a[0] != b[0]:
        print("Warning: a[0] != b[0]. First vector (toeplitz column) wins")

    t = b.copy()
    t[0] = a[0]
    for i in range(1, m):
        rot = np.hstack((a[i::-1], b[1:-i]))
        t = np.vstack((t, rot))

    return t
