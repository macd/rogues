def skewpart(a):
    """
    skewpart  Skew-symmetric (skew-Hermitian) part.
          skewpart(a) is the skew-symmetric (skew-Hermitian) part of A,
          (A - A')/2.
          It is the nearest skew-symmetric (skew-Hermitian) matrix to A in
          both the 2- and the Frobenius norms.
    """
    s = (a - a.T) / 2.

    return s
