def symmpart(a):
    """
    SYMMPART  Symmetric (Hermitian) part.
          SYMMPART(A) is the symmetric (Hermitian) part of A, (A + A')/2.
          It is the nearest symmetric (Hermitian) matrix to A in both the
          2- and the Frobenius norms.
    """
    s = (a + a.T) / 2.
    return s
