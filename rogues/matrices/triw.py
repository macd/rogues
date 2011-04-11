import numpy as np


class Higham(Exception):
    pass


def triw(n, alpha=-1, k=0):
    """
    triw   Upper triangular matrix discussed by Wilkinson and others.
       triw(n, alpha, k) is the upper triangular matrix with ones on
       the diagonal and ALPHAs on the first K >= 0 superdiagonals.
       n may be a 2-tuple, in which case the matrix is n[0]-by-n[1] and
       upper trapezoidal.
       Defaults: alpha = -1,
                 k = n - 1     (full upper triangle).
       triw(n) is a matrix discussed by Kahan, Golub and Wilkinson.

       Ostrowski (1954) shows that
         COND(TRIW(N,2)) = COT(PI/(4*N))^2,
       and for large ABS(ALPHA),
         COND(TRIW(N,ALPHA)) is approximately ABS(ALPHA)^N*SIN(PI/(4*N-2)).

       Adding -2^(2-N) to the (N,1) element makes TRIW(N) singular,
       as does adding -2^(1-N) to all elements in the first column.

       References:
       G.H. Golub and J.H. Wilkinson, Ill-conditioned eigensystems and the
          computation of the Jordan canonical form, SIAM Review,
          18(4), 1976, pp. 578-619.
       W. Kahan, Numerical linear algebra, Canadian Math. Bulletin,
          9 (1966), pp. 757-801.
       A.M. Ostrowski, On the spectrum of a one-parametric family of
          matrices, J. Reine Angew. Math., 193 (3/4), 1954, pp. 143-160.
       J.H. Wilkinson, Singular-value decomposition---basic aspects,
          in D.A.H. Jacobs, ed., Numerical Software---Needs and Availability,
          Academic Press, London, 1978, pp. 109-135.
    """
    try:
        m, n = n
    except TypeError:
        m = n

    if k == 0:
        k = n - 1

    if np.array(alpha).size != 1:
        raise Higham("Second Argument Must Be A Scalar.")

    t = np.tril(np.eye(m, n) + alpha * np.triu(np.ones((m, n)), 1), k)

    return t
