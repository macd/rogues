"""
The "utils" module is a reimplmentation of a few of Nick Higham's "Matrix
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
"""

from augment import *
from bandred import *
from cgs import *
from cond import *
from condeig import *
from cpltaxes import *
from dual import *
from ge import *
from hankel import *
from house import *
from mdsmax import *
from mgs import *
from pow2 import *
from ps import *
from repmat import *
from rq import *
from skewpart import *
from sparsify import *
from sub import *
from symmpart import *
from toeplitz import *
from treshape import *
from tridiag import *
from vand import *
from vecperm import *
