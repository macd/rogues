import numpy as np


class Higham(Exception):
    pass


def house(x):
    """
    house(x)   Householder matrix.
        If
           v, beta = house(x) then
           h  = eye - beta* np.outer(v, v)  is a Householder matrix such that
           Hx = -sign(x[0]) * norm(x) * e_1

        NB: If x = 0 then v = 0, beta = 1 is returned.
            x can be real or complex.
            sign(x) := exp(i*arg(x)) ( = x / abs(x) when x ~= 0).

        Theory: (textbook references Golub & Van Loan 1989, 38-43;
                 Stewart 1973, 231-234, 262; Wilkinson 1965, 48-50;
                 Higham 2002, 354-355)
        Hx = y: (I - beta*v*v')x = -s*e_1.
        Must have |s| = norm(x), v = x+s*e_1, and
        x'y = x'Hx =(x'Hx)' real => arg(s) = arg(x(1)).
        So take s = sign(x(1))*norm(x) (which avoids cancellation).
        v'v = (x(1)+s)^2 + x(2)^2 + ... + x(n)^2
            = 2*norm(x)*(norm(x) + |x(1)|).

        References:
        G.H. Golub and C.F. Van Loan, Matrix Computations, Third edition,
           Johns Hopkins University Press, Baltimore, Maryland, 1996.
        G.W. Stewart, Introduction to Matrix Computations, Academic Press,
           New York, 1973,
        J.H. Wilkinson, The Algebraic Eigenvalue Problem, Oxford University
           Press, 1965.
        N.J. Higham, Accuracy and Stability of Numerical Algorithms, SIAM,
           Philadelphia, 2002
    """
    if len(x.shape) == 1:
        n, = x.shape
    elif len(x.shape) == 2:
        n, m = x.shape
        if m != 1:
            raise Higham("x must be a vector")
    else:
        raise Higham("x must be a vector")

    # In numpy sign(0) = 0.0 and it looks like that might be the same in
    # m*lab.  Here we need sign(0) to be 1, rather than zero, which is the
    # reason for the code below.
    #
    sg = 1.0
    if np.sign(x[0]) < 0:
        sg = np.sign(x[0])

    s = sg * np.linalg.norm(x)
    v = x.copy()            # must copy or else x[0] get creamed

    # Quit if x is the zero vector.
    if s == 0:
        beta = 1
        return

    v[0] += s

    try:
        s = s.conjugate()
    except AttributeError:
        pass

    beta = 1 / (s * v[0])                           # NB the conjugated s.

    # beta = 1/(abs(s)*(abs(s)+abs(x(1)) would guarantee beta real.
    # But beta as above can be non-real (due to rounding) only when
    # x is complex.

    return v, beta, s
