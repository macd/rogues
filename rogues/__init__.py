"""
The "rogues" module is a reimplementation of Prof. N. Higham's test matrices into
Python, Numpy, and Scipy.  They were generally ported by using the
iPython shell.  They were developed from Version 3 (1995) of the test
matrix package from code downloaded from Higham's web site.  Also,
the earlier TOMS 694 version was used as reference in some cases.
That code was download from netlib.  Both of these packages were
accessed in February of 2009.

Also included are a small number of routines from Prof. Higham's matrixcomp
library as well as several required functions that had no implementations.

Some of the issues in porting to numpy from m*lab
    * numpy arrays have zero based array indexing while m*lab has one's
      based array indexing
    * numpy / python has indexing that does not include the upper end of
      the specified range, e.g. a[0:n]  is  a[0], a[1], ..., a[n-1].  This
      is different from m*lab
    * Of course, it is much easier to handle default values on the input
      parameters in Python
    * Element by element operation is the default in numpy.  To get the
      matrix behavior, the array must be converted to matrices.  Also,
      when dealing with arrays, we do not need to use the dot notation of
      m*lab  (ie x./y).  Also numpy has the a concept called broadcasting
      so that we can write and expression such as 1/x  which, if x is a
      array becomes  [[1/x[0,0], 1/x[0,1], ... rather than ones(n,n)./x
    * Some of the numpy functions take tuples for the shapes of arrays
      (notably zeros, ones, etc) while others do not (random.randn())
    * The m*lab routines that take matrix size arguments generally assume
      that a single dimension, say n, means the matrix is square, say n by n.
      This means that when you want a vector, you have to give the function
      _two_ arguments ie say zeros(n,1) or ones(1,n) etc. In numpy, one
      dimension is the default and we use zeros(n) etc.  When we need a
      two dimensional array we use zeros((m,n))
      
Comments and references were mostly preserved in the functions.  They were
slightly updated to reflect the changes necessary in Python

The inluded matrix generation functions are:

   cauchy
   chebspec
   chebvand
   chow
   clement
   comp
   compan
   condex
   cycol
   dingdong
   dorr
   dramadah
   fiedler
   forsythe
   frank
   gearm
   gfpp
   grcar
   hadamard
   hankel
   hanowa
   hilb
   invhess
   invol
   ipjfact
   jordbloc
   kahan
   kms
   krylov
   lauchli
   lehmer
   lesp
   lotkin
   minij
   moler
   neumann
   ohess
   parter
   pascal
   pdtoep
   pei
   pentoep
   poisson
   prolate
   qmult
   rando
   randsvd
   redheff
   riemann
   rogues
   smoke
   test
   triw
   wathen
   wilk
   wilkinson

Some of generally useful matrix utility functions:

   augment
   bandred
   cgs
   cond
   condeig
   cpltaxes
   dual
   ge
   hankel
   house
   mdsmax
   mgs
   pow2
   ps
   repmat
   rq
   skewpart
   sparsify
   sub
   symmpart
   toeplitz
   treshape
   tridiag
   vand
   vecperm

   More information is available on any of these functions by typing
   "help <funcname>"
   
Don MacMillen 1 August 2018
"""
name = "rogues"

__version__ = "0.5.0"
   
from rogues.matrices import *
from rogues.utils import *

