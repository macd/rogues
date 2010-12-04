"""
The "utils" module is a reimplmentation a few of Nick Higham's "Matrix
Computation Toolbox" [1], into Python, Numpy, and Scipy.  They were
generally ported using the iPython shell and probably work best there.
They were developed from version 1.2 (released on 5-Sep-2002 and downloaded
17 Feb 2009.

Also included in this package some are the generic matrix manipulation
functions that were inclued in Version 3, (1995 downloaded Feb 2009)
of Test Matrix package [2]


     [1] "N. J. Higham. The Matrix Computation Toolbox.
           http://www.ma.man.ac.uk/~higham/mctoolbox"
     [2] "N.J. Higham, The Test Matrix Toolbox for Matlab
           (Version 3.0) , NA Report 276, September 1995.
           http://www.maths.manchester.ac.uk/~higham/papers/misc.php "
     [3] "N.J. Higham, Algorithm 694: A collection of test matrices
           in MATLAB, ACM Trans. Math. Soft., 17(3):289--305, Sept. 1991. 

Don MacMillen 15 May 2009
matrix@macmillen.net
"""

import numpy as np
import numpy.linalg as nl
import numpy.random as nrnd
import scipy as sp
import scipy.linalg as sl
import scipy.sparse as sparse

class Higham(Exception):
    pass

def augment(a, alpha = 1.):
    """
    AUGMENT  Augmented system matrix.
         AUGMENT(A, ALPHA) is the square matrix
         [alpha*eye(m) a; a' zeros(n)] of dimension m+n, where a is m-by-n.
         It is the symmetric and indefinite coefficient matrix of the
         augmented system associated with a least squares problem
         minimize NORM(A*x-b).  alpha defaults to 1.
         Special case: if "a" is a scalar, n say, then augment(a) is the
                       same as AUGMENT(RANDN(p,q)) where n = p+q and
                       p = ROUND(n/2), that is, a random augmented matrix
                       of dimension n is produced.
         The eigenvalues of AUGMENT(A) are given in terms of the singular
         values s(i) of A (where m>n) by
                  1/2 +/- SQRT( s(i)^2 + 1/4 ),  i=1:n  (2n eigenvalues),
                  1,  (m-n eigenvalues).
         If m < n then the first expression provides 2m eigenvalues and the
         remaining n-m eigenvalues are zero.

         See also SPAUGMENT.

         Reference:
         G.H. Golub and C.F. Van Loan, Matrix Computations, Second
         Edition, Johns Hopkins University Press, Baltimore, Maryland,
         1989, sec. 5.6.4.
    """

    try:
        m, n  = a.shape
    except AttributeError:
        # Handle the special case a = scalar
        n = a
        p = np.round(n/2)
        q = n - p
        a = np.random.randn(p, q)
        m = p
        n = q

    c = np.vstack( (np.hstack((alpha * np.eye(m), a)),  np.hstack((a.T, np.zeros((n,n)))) ) )

    return c

def bandred(x, kl = 0, ku = -1):
    """
    BANDRED  Band reduction by two-sided unitary transformations.
         B = BANDRED(A, KL, KU) is a matrix unitarily equivalent to A
         with lower bandwidth KL and upper bandwidth KU
         (i.e. B(i,j) = 0 if i > j + KL or j > i + KU).
         The reduction is performed using Householder transformations.
         If KU is omitted it defaults to KL.

         Called by RANDSVD.
         This is a `standard' reduction.  Cf. reduction to bidiagonal form
         prior to computing the SVD.  This code is a little wasteful in that
         it computes certain elements which are immediately set to zero!

         Reference:
         G.H. Golub and C.F. Van Loan, Matrix Computations, second edition,
         Johns Hopkins University Press, Baltimore, Maryland, 1989.
         Section 5.4.3.
    """
    # Work on a local copy of the array.
    a = x.copy()
    
    if kl == 0 and (ku == 0 or ku == -1):
        raise Higham('You''ve asked for a diagonal matrix.  In that case use the SVD!')
    elif ku == -1:
        # set ku to k1 if it has not yet been set
        ku = kl

    # Check for special case where order of left/right transformations matters.
    # Easiest approach is to work on the transpose, flipping back at the end.
    flip = 0
    if ku == 0:
        a    = a.T
        kl, ku = ku, kl
        flip = 1

    m, n = a.shape

    for j in range( min( min(m,n), max(m-kl-1, n-ku-1) )):

        if j + kl + 1 <= m:
            v, beta, s = house( a[j+kl:m, j] )
            temp = a[j+kl:m, j:n]
            a[j+kl:m, j:n] = temp - beta * np.outer(v, np.dot(v, temp))
            a[j+kl+1:m,j]  = np.zeros(m-j-kl-1)

        if j + ku + 1 <= n:
            v, beta, s = house( a[j, j+ku:n] )
            temp = a[j:m, j+ku:n]
            a[j:m,j+ku:n] = temp - beta*np.outer(np.dot(temp, v), v)
            a[j,j+ku+1:n] = np.zeros(n-j-ku-1)

    if flip == 1:
        a = a.T

    return a

def cgs(a):
    """
    CGS     Classical Gram-Schmidt QR factorization.
    [Q, R] = cgs(A) uses the classical Gram-Schmidt method to compute the
    factorization A = Q*R for m-by-n A of full rank,
    where Q is m-by-n with orthonormal columns and R is n-by-n.

    NOTE: This was ported just for an example of the Gram-Schmidt orthogonalization.
    If you really want the QR factorization, you should probably use numpy.linalg.qr
    """
    m, n = a.shape
    q = np.zeros((m,n))
    r = np.zeros((n,n))

    # Gram-Schmidt expressed in matrix-vector form.
    # Treat j = 0 as special case to avoid if tests inside loop.
    r[0, 0] = nl.norm(a[:, 0])
    q[:, 0] = a[:, 0] / r[0, 0]
    for j in range(1, n):
        r[0:j-1, j] = np.dot(q[:, 0:j-1].T, a[:, j])            
        temp        = a[:, j] - np.dot(q[:, 0:j-1], r[0:j-1, j])
        r[j, j]     = nl.norm(temp)
        q[:,j]      = temp / r[j, j]

    return q, r

def chop(x, d = 9):
    """
    Just a very simple wrapper around np.around(),  We set the default number of
    decimal digits to round to at 9.
    """
    return np.around(x, d)

def cond(a, p = 2):
    """
    COND   Matrix condition number in 1, 2, Frobenius, or infinity norm.
       For p = 1, 2, 'fro', inf,  COND(A,p) = NORM(A,p) * NORM(INV(A),p).
       If p is omitted then p = 2 is used.
       A may be a rectangular matrix if p = 2; in this case COND(A)
       is the ratio of the largest singular value of A to the smallest
       (and hence is infinite if A is rank deficient).

       See also RCOND, NORM, CONDEST, NORMEST.

       This replicates (essentially) np.linalg.cond
    """
    
    if len(a) == 0:  # Handle null matrix.
        y = np.NaN
        return

    m, n = a.shape
    if m != n and p != 2:
        raise Higham('a is rectangular.  Use the 2 norm.')

    if p == 2:
        u, s, v = np.linalg.svd(a)
        if (s == 0).any():   # Handle singular matrix
            print 'Condition is infinite'
            y = np.Inf
        y = max(s) / min(s)
    else:
        #  We'll let NORM pick up any invalid p argument.
        y = np.linalg.norm(a, p) * np.linalg.norm(np.linalg.inv(a), p)

    return y

def condeig(a):
    """
    v, lambda, c = condeig(a) Computes condition numbers for the
    eigenvalues of a matrix. The condition numbers are the reciprocals
    of the cosines of the angles between the left and right eigenvectors.
    Inspired by Arno Onken's Octave code for condeig

    When checking against results obtained in Higham & Higham
    a = rogues.frank(6)
    lr, vr, c = matrixcomp.condeig(a)
    
    H & H get for lr  = [ 12.9736, 5.3832, 1.8355,  0.5448,  0.0771,  0.1858]
          and for  c  = [  1.3059, 1.3561, 2.0412, 15.3255, 43.5212, 56.6954]

    which they say that the small eigenvalues are slightly ill conditioned.
    With the this python/numpy condeig we get

    vr = [ 12.97360792, 5.38322318, 1.83552324, 0.54480378,   0.07707956,   0.18576231]
     c = [  1.30589002, 1.35605093, 2.04115713, 15.32552609, 43.52124194,  56.69535399]

    NOTE: we must use scipy.linalg.decomp.eig and _not_ np.linalg.eig
    """

    if len(a.shape) != 2:
        raise Higham("a must be a 2 dimensional array")
    else:
        m, n = a.shape
        if m != n or m < 2:
            raise Higham("a must be a square array with dimension 2 or greater")

    # eigenvalues, left and right eigenvectors
    lamr, vl, vr = sl.eig(a, left = True, right = True)
    
    # Need to put the left eigenvectors into row form
    vl = vl.T
    
    # Normalize vectors
    for i in range(n):
        vl[i,:] = vl[i,:] / np.sqrt( abs(vl[i,:]**2).sum() )

    # Condition numbers are reciprocal of the cosines (dot products) of the
    # left eignevectors with the right eigenvectors.   In a perfect world,
    # these numbers should all be one, but they are not.  
    c = abs (1 / np.diag(np.dot(vl, vr)) )

    return lamr, vr, c

def dual(x, p = None):
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

    if p == None:
        if len(x) == 1:
            y = 1/(1-1/x)
            return
        else:
            raise Higham('Second argument missing.')

    q = 1 / (1 - 1/float(p))

    if nl.norm(x, np.inf) == 0:
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
        x = x / nl.norm(x, np.inf)         # This scaling helps to avoid under/over-flow.
        y = np.abs(x)**(p-1) * np.where(np.sign(x)==0, 1, np.sign(x))
        y = y / nl.norm(y, q)         # Normalize to unit q-norm.

    return y

def ge(b):
    """
    GE     Gaussian elimination without pivoting.
       [l, u, rho] = ge(a) computes the factorization a = l*u,
       where L is unit lower triangular and U is upper triangular.
       RHO is the growth factor.
       By itself, ge(a) returns the final reduced matrix from the
       elimination containing both L and U.

       Note added in porting to Python/numpy/scipy:
       --------------------------------------------
       There are obviously more efficient routines in numpy / scipy
       but this routine is intended for testing numerical properties,
       ie it can be used in the direct search method adsmax for finding
       a matrix that maximizes the value of rho. See the following very
       readable and fun paper that is available from Prof. Higham's web
       site:

        Reference:
        N.J. Higham, Optimization by direct search in matrix computations,
        SIAM J. Matrix Anal. Appl, 14(2): 317-333, April 1993.
       
    """
    a = b.copy()    # don't cream the input matrix
    n, n = a.shape
    maxA = a.max()
    rho = maxA

    for k in range(n - 1):

        if a[k,k] == 0:
            raise Higham('Elimination breaks down with zero pivot.')

        a[k+1:n,k] = a[k+1:n,k] / a[k,k]          # Multipliers.

        # Elimination
        a[k+1:n, k+1:n] = a[k+1:n, k+1:n] - np.outer(a[k+1:n, k], a[k, k+1:n])
        rho = max( rho, (abs(a[k+1:n, k+1:n])).max()  )

    l = np.tril(a, -1) + np.eye(n)
    u = np.triu(a)
    rho = rho/maxA

    return l, u, rho

def hankel(a, b = None):
    """
    hankel(a) returns a toeplitz matrix given "a", the first row of the
    matrix.  This matrix is defined as:

         [[   a[0], a[1], a[2], a[3], ..., a[n-1], a[n] ]]
          [   a[1], a[2], a[3], a[4], ...,   a[n],   0  ]
          [   a[2], a[3], a[4], ...,     ]
          ...
          [ a[n-2], a[n-1], a[n],  0,   ...
          [ a[n-1], a[n],      0,  0,   ...           0
          [   a[n],    0,      0,  0,   ...           0 ] ]


    Note that all the non-zero anti-diagonals are constant

    If called as hankel(a, b) then create the hankel matrix where a is the
    first column and b is the last row.  If a[-1] != b[0], a[-1] is chosen
    but a warning message is printed.  For example

    In [2]: from hankel import *
    In [3]: a = arange(10)
    In [4]: a
    Out[4]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    In [5]: b = arange(10,20)
    In [6]: h = hankel(a, b)
    Warning: a[-1] != b[0]. a[-1] is chosen
    In [7]: h
    Out[7]:
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
           [ 2,  3,  4,  5,  6,  7,  8,  9, 11, 12],
           [ 3,  4,  5,  6,  7,  8,  9, 11, 12, 13],
           [ 4,  5,  6,  7,  8,  9, 11, 12, 13, 14],
           [ 5,  6,  7,  8,  9, 11, 12, 13, 14, 15],
           [ 6,  7,  8,  9, 11, 12, 13, 14, 15, 16],
           [ 7,  8,  9, 11, 12, 13, 14, 15, 16, 17],
           [ 8,  9, 11, 12, 13, 14, 15, 16, 17, 18],
           [ 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
    """

    # Error checking...
    try:
        m,  = a.shape
        if b != None:
            n, = b.shape
    except (ValueError, AttributeError):
        raise Higham("Input arrays must be one dimensional")

    if b == None:
        b = np.zeros_like(a)
        n = m
    elif a[-1] != b[0]:
        print("Warning: a[-1] != b[0]. a[-1] is chosen")

    k = np.r_[a, b[1:]]
    h = k[:n]
    for i in xrange(1,m):
        h = np.vstack((h, k[i:i+n]))
    
    return h

def house(x):
    """
    house(x)   Householder matrix.
        If
           v, beta = house(x)             then
           h  = eye - beta* np.outer(v, v)            is a Householder matrix such that
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
        n,m = x.shape
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
        
    s = sg * nl.norm(x)
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
    # But beta as above can be non-real (due to rounding) only when x is complex.

    return v, beta, s

def mgs(a):
    """
    MGS     Modified Gram-Schmidt QR factorization.
        q, r = mgs(a) uses the modified Gram-Schmidt method to compute the
        factorization a = q*e for m-by-n a of full rank,
        where q is m-by-n with orthonormal columns and R is n-by-n.
    """
    try:
        m, n = a.shape
    except AttributeError:
        raise Higham("Input array must be two dimensional")
    
    q = np.zeros((m,n))
    r = np.zeros((n,n))

    for k in range(n):
        r[k, k]     = nl.norm(a[:, k])
        q[:, k]     = a[:,k] / r[k,k]
        r[k, k+1:n] = np.dot(q[:,k], a[:,k+1:n])
        a[:, k+1:n] = a[:,k+1:n] - np.outer(q[:,k], r[k,k+1:n])

    return q, r

def pow2(x, y = None):
    """
    Raise 2 to the power of x[i] for the vector x.  If two vectors
    are supplied, return    x[i] * (2 ** y[i])
    Note that no error checking is done in this example.
    """
    if y == None:
        z = (2.*np.ones(len(x))) ** x
    else:
        z = x * ( (2.*np.ones(len(y))) ** y)
        
    return z

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
        b = np.hstack((b,a))

    c = b
    for i in range(1, m):
        c = np.vstack((c,b))

    return c

def rq(A,x):
    """
    rg      Rayleigh quotient.
            rq(a, x) is the Rayleigh quotient of a and x, x'*A*x/(x'*x).

        Called by FV.
        NOTE: This function has a name clash with scipy.linalg.rq which
        computes the RQ decomposition of a matrix.
    """
    z = np.dot(x, np.dot(A,x)) / np.dot(x,x)

    return z

def toeplitz(a, b = None):
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
        m,  = a.shape
        if b != None:
            n, = b.shape
    except (ValueError, AttributeError):
        raise Higham("Input arrays must be one dimensional")

    if b != None:
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
    for i in xrange(1,m):
        rot = np.r_[a[i::-1],  b[1:-i]]
        t = np.vstack((t, rot))
    
    return t

def tridiag(n, x = None, y = None, z = None):
    """
    tridiag  tridiagonal matrix (sparse).
         tridiag(x, y, z) is the sparse tridiagonal matrix with 
         subdiagonal x, diagonal y, and superdiagonal z.
         x and z must be vectors of dimension one less than y.
         Alternatively tridiag(n, c, d, e), where c, d, and e are all
         scalars, yields the toeplitz tridiagonal matrix of order n
         with subdiagonal elements c, diagonal elements d, and superdiagonal
         elements e.   This matrix has eigenvalues (todd 1977)
                  d + 2*sqrt(c*e)*cos(k*pi/(n+1)), k=1:n.
         tridiag(n) is the same as tridiag(n,-1,2,-1), which is
         a symmetric positive definite m-matrix (the negative of the
         second difference matrix).

         References:
         J. Todd, Basic Numerical Mathematics, Vol. 2: Numerical Algebra,
           Birkhauser, Basel, and Academic Press, New York, 1977, p. 155.
         D.E. Rutherford, Some continuant determinants arising in physics and
           chemistry---II, Proc. Royal Soc. Edin., 63, A (1952), pp. 232-241.
    """
    #
    # I'm not too happy with the way I've handled the mallable nature of
    # the function signitures... this is what function overloading is
    # supposed to do.
    #
    try:
        # First see if they are arrays
        nx, = n.shape
        ny, = x.shape
        nz, = y.shape
        if (ny - nx - 1) != 0 or  (ny - nz -1) != 0:
            raise Higham('Dimensions of vector arguments are incorrect.')
        # Now swap to match above
        z = y
        y = x
        x = n

    except AttributeError:
        # They are not arrays
        if n < 2:
            raise Higham("n must be 2 or greater")
        
        if x == None and y == None and z == None:
            x = -1
            y =  2
            z = -1
            
        x = x * np.ones(n - 1)
        z = z * np.ones(n - 1)
        y = y * np.ones(n)

    except ValueError:
        raise Higham("x, y, z must be all scalars or 1-D vectors")

    # t = diag(x, -1) + diag(y) + diag(z, 1);  % For non-sparse matrix.
    n  = np.max(np.size(y))
    za = np.zeros(1)

    # Use the (*)stack functions instead of the r_[] notation in
    # an attempt to be more readable
    t = sp.sparse.spdiags(np.vstack( ( np.hstack((x, za)),
                                                         y,
                                          np.hstack((za, z))) ),
                             np.array([-1,0,1]), n, n);

    return t

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

def sparsify(a, p = 0.25):
    """
    SPARSIFY  Randomly set matrix elements to zero.
          S = SPARSIFY(A, P) is A with elements randomly set to zero
          (S = S' if A is square and A = A', i.e. symmetry is preserved).
          Each element has probability P of being zeroed.
          Thus on average 100*P percent of the elements of A will be zeroed.
          Default: P = 0.25.

          Note added in porting: by inspection only, it appears the the m*lab
          version may have a bug where it always returns zeros on the diagonal
          for a symmetric matrix... can anyone confirm?
     """

    if p < 0 or p > 1:
        raise Higham('Second parameter must be between 0 and 1 inclusive.')

    m, n = a.shape

    if (a == a.T).all():
        # Preserve symmetry
        d = np.choose(nrnd.rand(m) > p, (np.zeros(m), np.diag(a)))
        a = np.triu(a, 1) * (nrnd.rand(m, n) > p)   
        a = a + a.T
        a = a + np.diag(d)
    else:
        # Unsymmetric case
        a = np.choose(nrnd.rand(m,n) > p, ( np.zeros((m,n)), a) )               

    return a

def sub(a, i, j = None):
    """
    SUB     Principal submatrix.
        sub(a,i,j) is a[i:j, i:j].
        sub(a,i)  is the leading principal submatrix of order i,
        a[0:i, 0:i), if i > 0, and the trailing principal submatrix
        of order ABS(i) if i<0.
    """
    # editorial comment: this function seems unnecessary... it strikes
    # me that you would always want the bare code rather than the function
    # call because it is self evident what it does
    if j == None:
        if i >= 0:
            s = a[0:i, 0:i]
        else:
            n = np.min(a.shape)
            s = a[n+i:n, n+i:n]

    else:
        s = a[i:j, i:j]

    return s

def symmpart(a):
    """
    SYMMPART  Symmetric (Hermitian) part.
          SYMMPART(A) is the symmetric (Hermitian) part of A, (A + A')/2.
          It is the nearest symmetric (Hermitian) matrix to A in both the
          2- and the Frobenius norms.
    """
    s = (a + a.T)/2.
    return s

def treshape(x, unit = 0, row_wise = False):
    """
    treshape  reshape vector to or from (unit) triangular matrix.
          treshape(x) returns a square upper triangular matrix whose
          elements are taken columnwise from the matrix x.
          If row_wise = True, then the matrix elements are taken
          row wise from the matrix x.
          treshape(x,1) returns a UNIT upper triangular matrix, and
          the 1's should not be specified in X.
          An error results if X does not have a number of elements of the form
          n*(n+1)/2 (or n less than this in the unit triangular case).
          x = treshape(r,2) is the inverse operation to r = treshape(x).
          x = treshape(r,3) is the inverse operation to r = treshape(x,1).
    """

    try:
        p, q = x.shape
    except ValueError:
        p, = x.shape
        q = 1
        
    if unit < 2:   # Convert vector x to upper triangular R.

        m = p * q
        n = int(np.around( (-1 + np.sqrt(1 + 8*m))/2. ))
        if n*(n+1)/2 != m:
            raise Higham('Matrix must have a ''triangular'' number of elements.')

        if unit == 1:
            n = n + 1

        x = x.ravel()
        t = unit * np.eye(n)

        i = 0
        if row_wise:
            for j in range(n - unit):
                t[j, j+unit:n] = x[i : i + n - unit - j]
                i = i + n - unit - j
                
        else:
            for j in range(unit,n):
                t[0:j-unit+1, j] = x[i:i+j-unit+1]
                i = i + j - unit + 1

    elif unit >= 2:   # Convert upper triangular R to vector x.

        t = x
        if p != q:
            raise Higham('Must pass square matrix')
        
        unit = unit - 2
        n = p*(p+1)/2 - unit * p
        x = np.zeros(n)
        i = 0
        if row_wise:
            for j in range(p - unit):
                x[i : i + p - unit - j] = t[j, j+unit:p]
                i = i + p - unit - j

        else:
            for j in range(unit, p):
                x[i:i+j-unit+1] = t[0:j-unit+1, j]
                i = i + j - unit + 1

        t = x

    return t

def vand(p, m = 0):
    """
    VAND   Vandermonde matrix.
       v = vand(p), where p is a vector, produces the (primal)
       Vandermonde matrix based on the points p, i.e. v[i,j] = p[j]**(i-1)
       vand(p, m) is a rectangular version of vand(p) with m rows
       Special case: If P is a scalar then P equally spaced points on [0,1]
                     are used.

       Reference:
       N. J. Higham, Accuracy and Stability of Numerical Algorithms,
       Second edition, Society for Industrial and Applied Mathematics,
       Philadelphia, PA, 2002; chap. 22

       Notes added in porting:
       You should probably use np.vander instead of this function, but
       then the interface is different and you get the row reversed
       transpose of the matrix ( ie a.T[::-1,:] ) returned by vand
       
       WARNING... changed the function signiture from the Higham m*lab version
       by reversing the order of the arguments.
    """
    try:
        n = len(p)
    except TypeError:
        #  Handle scalar p.
        n = p
        p = np.linspace(0, 1, n)

    if m == 0:
        m = n

    v = np.ones((m,n))
    for i in range(1, m):
        v[i, :] = p * v[i-1, :]

    return v

def vecperm(m, n = None):
    """
    VECPERM    Vec-permutation matrix.
           VECPERM(M, N) is the vec-permutation matrix, an MN-by-MN
           permutation matrix P with the property that if A is M-by-N then
           vec(A) = P*vec(A').
           If N is omitted, it defaults to M.

           P is formed by taking every n'th row from EYE(M*N), starting with
           the first and working down - see p. 277 of the reference.

           Reference:
           H. V. Henderson and S. R. Searle The vec-permutation matrix,
           the vec operator and Kronecker products: A review Linear and
           Multilinear Algebra, 9 (1981), pp. 271-288.
    """
    if n == None:
        n = m

    p = np.zeros((m*n, m*n))
    e = np.eye(m*n)

    k = 0
    for i in range(n):
       p[k:k+n, :] = e[i:m*n:n, :]
       k += n

    return p
