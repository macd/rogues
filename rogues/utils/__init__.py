"""
The "utils" module is a reimplmentation of a few of Nick Higham's "Matrix
Computation Toolbox" [1], into Python, Numpy, and Scipy.  They were
generally ported using the iPython shell and probably work best there.
They were developed from version 1.2 (released on 5-Sep-2002 and downloaded
17 Feb 2009.)

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

from rogues.utils.augment import *
from rogues.utils.bandred import *
from rogues.utils.cgs import *
from rogues.utils.cond import *
from rogues.utils.condeig import *
from rogues.utils.cpltaxes import *
from rogues.utils.dual import *
from rogues.utils.fv import *
from rogues.utils.ge import *
from rogues.utils.gersh import *
# duplicate, don't use this one
# from rogues.utils.hankel import *
from rogues.utils.house import *
from rogues.utils.mdsmax import *
from rogues.utils.mgs import *
from rogues.utils.pow2 import *
from rogues.utils.ps import *
from rogues.utils.pscont import *
from rogues.utils.repmat import *
from rogues.utils.rq import *
from rogues.utils.skewpart import *
from rogues.utils.sparsify import *
from rogues.utils.sub import *
from rogues.utils.symmpart import *
from rogues.utils.toeplitz import *
from rogues.utils.treshape import *
from rogues.utils.tridiag import *
from rogues.utils.vand import *
from rogues.utils.vecperm import *
