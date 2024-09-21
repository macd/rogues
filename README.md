# rogues 1.0.0

## Python and Numpy port of Prof. Nicholas Higham's matlab test matrices

These matrices are a collection of interesting matrices that appear in
matlab's 'gallery' collection. This collection was originally defined
and implemented by Prof. Nicholas Higham of Manchester University and
is more fully discussed in "The Test Matrix Toolbox for Matlab
(Version 3.0)", N.J. Higham, Numerical Analysis Report No. 276,
September 1995 and available [here](http://www.ma.man.ac.uk/~higham/mctoolbox/toolbox.pdf) (Now appears to be a dead link)

The pdf of the 1991 TOMS (Algorithm 694) is available from ACM [here](https://dl.acm.org/citation.cfm?id=116805)

By 'interesting' we mean that these matrices either present some
challenges to numerical algorithms or have some a set of interesting
properties. The documentation of the individual functions contains
much more info, as well as references.

Also included are a set of matrix utility functions that are needed
for generating some of members of the collection as well as a few
functions from Prof. Higham’s matrixcomp package. One of the more
interesting routines here is mdsmax, a direct search optimization
algorithm.

The rogues package depends on numpy and scipy, both of which must be
installed. Additionally, there are a few routines that deal with
plotting, and these use matplotlib. While ipython is not strictly
necessary, it is a very convenient environment for numpy / scipy /
matplotlib. Finally, the unit tests now utilize pytest, and that
must be installed in your environment. Then just make certain you
are in the rogues top level directory and run pytest, ie

    cd rogues
    pytest

The included matrix generation functions are:

* **cauchy** Cauchy matrix
* **chebspec** Chebyshev spectral differentiation matrix
* **chebvand** Vandermonde-like matrix for the Chebyshev polynomials
* **chow** Chow matrix - a singular Toeplitz lower Hessenberg matrix
* **clement** Clement matrix - tridiagonal with zero diagonal entries
* **comp** Comparison matrices
* **compan** Companion matrix
* **condex** Counterexamples to matrix condition number estimators
* **cycol** Matrix whose columns repeat cyclically
* **dingdong** Dingdong matrix - a symmetric Hankel matrix
* **dorr** Dorr matrix - diagonally dominant, ill conditioned, tridiagonal.
* **dramadah** A (0,1) matrix whose inverse has large integer entries
* **fiedler** Fiedler matrix - symmetric
* **forsythe** Forsythe matrix - a perturbed Jordan block
* **frank** Frank matrix - ill conditioned eigenvalues.
* **gearm** Gear matrix
* **gfpp** Matrix giving maximal growth factor for GW with partial pivoting
* **grcar** Grcar matrix - a Toeplitz matrix with sensitive eigenvalues.
* **hadamard** Hadamard matrix
* **hankel** Hankel matrix
* **hanowa** A matrix whose eigenvalues lie on a vertical line in C
* **hilb** Hilbert matrix
* **invhess** Inverse of an upper Hessenberg matrix
* **invol** An involutory matrix
* **ipjfact** A Hankel matrix with factorial elements
* **jordbloc** Jordan block matrix
* **kahan** Kahan matrix - upper trapezoidal
* **kms** Kar-Murdock-Szego Toeplitz matrix
* **krylov** Krylov matrix
* **lauchli** Lauchli matrix - rectangular
* **lehmer** Lehmer matrix - symmetric positive definite
* **lesp** A tridiagonal matrix with real, sensitve eigenvalues
* **lotkin** Lotkin matrix
* **minij** Symmetric positive definite matrix min(i,j)
* **moler** Moler matrix symmetric positive definite
* **neumann** Singular matrix from the descrete Neumann problem (sparse)
* **ohess** Random, orthogonal upper Hessenberg matrix
* **parter** Parter matrix - a Toeplitz matrix with singular values near pi
* **pascal** Pascal matrix
* **pdtoep** Symmetric positive definite Toeplitz matrix
* **pei** Pei matrix
* **pentoep** Tentadiagonal Toeplitz matrix (sparse)
* **poisson** Block tridiagonal matrix from Poisson’s equation (sparse)
* **prolate** Prolate matrix - symmetric, ill-conditioned Toeplitz matrix
* **qmult** Pre-multiply by random orthogonal matrix
* **rando** Random matrix with elements -1, 0, or 1
* **randsvd** Random matrix with pre-assigned singular values
* **redheff** A (0,1) matrix of Redheffer associated with the Riemann hypothesis
* **riemann** A matrix associated with the Riemann hypothesis
* **smoke** Smoke matrix - complex, with a ‘smoke ring’ pseudospectrum
* **triw** Upper triangular matrix discussed by Wilkinson and others
* **wathen** Wathen matrix - a finite element matrix (sparse, random entries)
* **wilk** Various specific matrices devised /discussed by Wilkenson
* **wilkinson** Wilkinson matrix of size n, where n must be odd

Some of generally useful matrix utility functions:

* **augment** Agumented system matrix
* **bandred** Band reduction by two-sided unitary transformations
* **cgs** Classical Gram-Schmidt QR factorization
* **cond** Matrix condition number in 1,2,Frobenius, or infinity norm
* **condeig** Condition numbers for eigenvalues of a matrix
* **cpltaxes** Determine suitable axis for plot of complex vector
* **dual** Dual vector with respect to Holder p-norm
* **fv**  Evaluate and plot the field of values largest leading submatrices
* **ge** Gaussian elimination without pivoting
* **gersh** Plots Gershgorin disks for a square matrix
* **hankel** Given first row, returns a Toeplitz type matrix
* **house** Householder matrix
* **mdsmax** Multidimensional search method for direct search optimization
* **mgs** Modified Gram-Schmidt QR factorization
* **pow2** Vector whose i-th element is 2 ** x[i], where x[] is input
* **ps** Dot plot of a pseudospectrum
* **pscont** Plots contours and color plots of pseudospectra
* **repmat** Simple re-implementation of matlab's repmat function
* **rq** Rayleigh quotient
* **skewpart** Skew-symmetric (skew-Hermitian) part
* **sparsify** Randomly sets matrix elements to zero
* **sub** Principal submatrix
* **symmpart** Symmetric (Hermitian) part
* **toeplitz** Returns toeplitz matrix given first row of the matrix
* **treshape** Reshape vector to or from (unit) triangular matrix
* **tridiag** Sparse tridiagonl matrix given the diagonals
* **vand** Vandermonde matrix
* **vecperm** Vector permutation matrix

More information is available on any of these functions by typing **help <funcname>**

### 0.5.0 Release Notes
    Don't use the distribute_setuptools.py stuff for python 2.7.15
    Fixed type errors in cycol.py, dorr.py, hanowa.py, ohess.py, and treshape.py that have
    developed with newer versions of Python.
    Tested on Ubuntu 18.04.1 with Python 2.7.15 & Numpy 1.14.3; Python 3.6.5 & Numpy 1.14.3;
        and Python 3.7.0 and Numpy 1.15.0
### 0.4.0 Release Notes
    Added visualization routines fv, gersh, and pscont. Fixed issues with np.r_.
    Relaxed tolerances for two problematic unit tests. Only use distribute with
    Python 2 series.  Cleaned up pep8 warnings.
    Tested on Ubuntu 14.04 with Python 3.5.1,  Numpy 1.10.2, Scipy 0.16.1, IPython 4.0.1, Matplotlib 1.5.0
    Tested on Ubuntu 14.04 with Python 3.4.3,  Numpy 1.9.3,  Scipy 0.16.0, IPython 4.0.0, Matplotlib 1.4.3
    Tested on Ubuntu 14.04 with Python 2.7.10, Numpy 1.9.2,  Scipy 0.15.1, IPython 3.2.1, Matplotlib 1.3.1
    Tested on Windows 10   with Python 3.5.1,  Numpy 1.10.1, Scipy 0.16.0, IPython 4.0.0, Matplotlib 1.4.3

### 1.0.0 Release Notes
    Simplified tests and moved to pytest
    Removed deprecation warnings
    Tested on Ubuntu 24.04, Python 3.8.19 with Numpy 1.24.4, and Python 3.12.5 with Numpy 2.1.0

### 0.3.0 Release Notes
    Ported to Python 3. Added distribute_setup.py to fix installation problems.
    Tested on Ubuntu 12.04 with Python 2.7.3, Numpy 1.6.1, Scipy 0.10.0, IPython 0.12
    Tested on Ubuntu 12.04 with Python 3.2.3, Numpy 1.6.2, Scipy 0.11.0, IPython 0.13.1
    Tested on Windows 7    with Python 2.7.3, Numpy 1.6.2, Scipy 0.11.0, IPython 0.13.1

### 0.2.0 Release Notes
    Unit tests now included with distribution.


