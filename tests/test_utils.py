import numpy as np
import numpy.linalg as nl
import numpy.random as nrnd
import numpy.testing as npt
import rogues


def test_augment():
    """One specific simple test for augment"""
    a = np.array([[1., 0., 0., 0., 4., 3., 2., 1.],
                  [0., 1., 0., 0., 3., 3., 2., 1.],
                  [0., 0., 1., 0., 0., 2., 2., 1.],
                  [0., 0., 0., 1., 0., 0., 1., 1.],
                  [4., 3., 0., 0., 0., 0., 0., 0.],
                  [3., 3., 2., 0., 0., 0., 0., 0.],
                  [2., 2., 2., 1., 0., 0., 0., 0.],
                  [1., 1., 1., 1., 0., 0., 0., 0.]])
    b = rogues.frank(4)
    c = rogues.augment(b)
    npt.assert_array_almost_equal(a, c, 12)


def test_bandred():
    """Simple Test of bandred, checks againts np.linalg.cond
    NEEDS MORE TESTING!  This test is pretty lame...
    """
    a = rogues.frank(9)
    # Use bandred to get a tri-diagonal form
    b = rogues.bandred(a, 1, 1)
    # Chop into a tri-diagonal form
    c = np.triu(np.tril(b, 1), -11)
    # They must be equal
    npt.assert_array_almost_equal(b, c)


def test_cgs():
    """Simple Test of cgs, classical Gram-Schmidt orthogonalization"""
    a = nrnd.rand(10, 10)
    q, r = rogues.cgs(a)
    # r must be upper triangular
    npt.assert_almost_equal(r, np.triu(r), 10)
    # factorization must hold
    npt.assert_almost_equal(a, q @ r, 10)


def test_cond():
    """Simple Test of cond, checks againts np.linalg.cond"""
    a = np.diag(np.arange(1, 11)) + np.diag(np.ones(9), 1)
    c1 = rogues.cond(a)
    c2 = np.linalg.cond(a)
    npt.assert_almost_equal(c1, c2, 10)


def test_condeig():
    """
    Simple Test of condeig, checks (partly) results in Higham & Higham
    Note that the last two condtion numbers do _not_ match the results
    of H&H and seem exceptionally large
    """
    a = rogues.frank(6)
    gold_c = [1.30589002, 1.35605093, 2.04115713, 15.32552609, \
              43.52124194, 56.69535399]
    lr, vr, c = rogues.condeig(a)

    for i in range(6):
        npt.assert_almost_equal(c[i], gold_c[i], 6)

# @npt.dec.knownfailureif(True, "diagpiv hasn't been debugged yet!")
# def test_diagpiv():
#     """Simple exercise of diagpiv"""
#     l,d,p,rho = rogues.diagpiv(5)
#     l,u,r = ge(a)
#     b = l @ u
#     npt.assert_array_almost_equal(a, b, 12)


def test_dual():
    """Simple Test of dual, checks the the product of the norms"""
    p = 3
    q = 1 / (1. - 1. / float(p))
    x = nrnd.rand(20)
    y = rogues.dual(x, p)
    npt.assert_almost_equal(np.dot(x, y), nl.norm(x, p) * nl.norm(y, q), 10)


def test_ge():
    """Simple exercise of the ge (Gaussian elimination)"""
    a = rogues.gfpp(5)
    l, u, r = rogues.ge(a)
    b = l @ u
    npt.assert_array_almost_equal(a, b, 12)

# tested in rogues/matrices/tests
# def test_hankel():
#     """
#     Simple test of hankel matrix.
#     """
#     a = np.arange(10)
#     b = np.arange(10, 20)
#     h = rogues.hankel(a, b)
#     g = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
#                   [2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
#                   [3, 4, 5, 6, 7, 8, 9, 11, 12, 13],
#                   [4, 5, 6, 7, 8, 9, 11, 12, 13, 14],
#                   [5, 6, 7, 8, 9, 11, 12, 13, 14, 15],
#                   [6, 7, 8, 9, 11, 12, 13, 14, 15, 16],
#                   [7, 8, 9, 11, 12, 13, 14, 15, 16, 17],
#                   [8, 9, 11, 12, 13, 14, 15, 16, 17, 18],
#                   [9, 11, 12, 13, 14, 15, 16, 17, 18, 19]])

#     npt.assert_array_equal(h, g)


def test_house():
    """
    Simple test of house (Householder transform). Just checks to
    see if the det(h) == -1, (reflection matrices have this property)
    """
    x = np.ones(5)
    v, beta, s = rogues.house(x)
    h = np.eye(5) - beta * np.outer(v, v)
    d = nl.det(h)
    npt.assert_almost_equal(d, -1., 12)


def test_house_reflect():
    """
    Simple test of house (Householder transform). Just checks to
    see if the input vector is reflected to be all in the x[0] direction
    """
    x = np.ones(5)
    v, beta, s = rogues.house(x)
    h = np.eye(5) - beta * np.outer(v, v)
    d = np.dot(h, x)
    d2 = np.dot(d, d)
    npt.assert_almost_equal(d2, d[0] ** 2, 12)


def test_mgs():
    """Simple test modified gram schmidt orthogonalization"""
    a = np.random.randn(10, 10)
    q, r = rogues.mgs(a)
    b = q @ q.T
    npt.assert_array_almost_equal(b, np.eye(10), 12)


def test_repmat():
    """
    Simple test of repmat. (repeated matrix)
    """
    a = np.array([2, 3, 4])
    b = rogues.repmat(a, 3, 4)
    c = np.array([[2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
                  [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4],
                  [2, 3, 4, 2, 3, 4, 2, 3, 4, 2, 3, 4]])
    npt.assert_array_equal(b, c)


def test_rq():
    """Simple test of the Raleigh quotient (x.(A.x))/(x.x)"""
    n = 10
    a = np.eye(n)
    b = np.ones(n)
    npt.assert_equal(rogues.rq(a, b), 1)


def test_skewpart():
    """Simple test of taking the skew part of a matrix"""
    a = nrnd.rand(10, 10)
    s = rogues.skewpart(a)
    npt.assert_array_almost_equal(s, -s.T)


def test_sparsify():
    """Simple test of sparsify'ing a matrix."""
    n = 1000
    prob = 0.25
    a = np.ones((n, n))
    s = rogues.sparsify(a, prob)
    assert(s.sum() > n * n * (1 - 1.05 * prob))


def test_sub():
    """Simple test of extracting the principal submatrix."""
    n = 10
    a = np.zeros((n, n))
    b = 5 * np.ones((3, 3))
    a[0:3, 0:3] = b
    s = rogues.sub(a, 3)
    npt.assert_array_equal(b, s)
    b = np.zeros((3, 3))
    b[0, 0] = 5
    s = rogues.sub(a, 2, 5)
    npt.assert_array_equal(b, s)


def test_symmpart():
    """Simple test of extracting the symmetric part of a matrix."""
    n = 10
    a = nrnd.rand(n, n)
    b = rogues.symmpart(a)
    e, w = nl.eig(b)
    z = np.zeros(n)
    npt.assert_array_equal(e.imag, z)


def test_toeplitz():
    """
    Simple test of toeplitz. This just checks that the
    known values of  determinant for a simple form at a
    succession of matrix sizes
    """
    dets = []
    for i in range(3, 30):
        a = np.arange(1, i)
        b = rogues.toeplitz(a)
        dets.append(nl.det(b))
    dets = np.array(dets)

    # Here are the known values of the determinants
    ans  = np.array([(-1) ** (i - 1) * (i + 1) * (2 ** (i - 2)) for i in \
                       range(2, 29)])

    # pretty bad round off for some values
    npt.assert_array_almost_equal(dets, ans, 5)


def test_treshape_0():
    """Simple test of treshape_0"""
    n = 4
    m = n * (n + 1) / 2
    a = np.arange(1, m + 1)
    b = rogues.treshape(a)
    c = rogues.treshape(b, unit=2)
    npt.assert_array_equal(a, c)


def test_treshape_1():
    """Simple test of treshape_1"""
    n = 4
    m = n * (n + 1) / 2
    a = np.arange(1, m + 1)

    d = rogues.treshape(a, unit=1)
    e = rogues.treshape(d, unit=3)
    npt.assert_array_equal(a, e)
    npt.assert_equal(nl.det(d), 1.0)


def test_treshape_2():
    """Simple test of treshape_2"""
    n = 4
    m = n * (n + 1) / 2
    a = np.arange(1, m + 1)
    b = rogues.treshape(a, row_wise=True)
    c = rogues.treshape(b, unit=2, row_wise=True)
    npt.assert_array_equal(a, c)


def test_treshape_3():
    """Simple test of treshape_3"""
    n = 4
    m = n * (n + 1) / 2
    a = np.arange(1, m + 1)

    d = rogues.treshape(a, unit=1, row_wise=True)
    e = rogues.treshape(d, unit=3, row_wise=True)
    npt.assert_array_equal(a, e)
    npt.assert_equal(nl.det(d), 1.0)


def test_tridiag_a():
    """Simple test of tridiag.  Just recover the diagonals"""
    n = 10           # size of array
    x = []
    x.append(np.ones(n - 1))
    x.append(2 * np.ones(n))
    x.append(3 * np.ones(n - 1))
    a = rogues.tridiag(x[0], x[1], x[2])
    b = a.todense()

    npt.assert_array_equal(x[0], np.diag(b, -1))
    npt.assert_array_equal(x[1], np.diag(b, 0))
    npt.assert_array_equal(x[2], np.diag(b, 1))


def test_tridiag_b():
    """Simple test of tridiag. Check the const diags from single arg"""
    n = 10
    a = rogues.tridiag(n)
    b = a.todense()
    x = []
    x.append(-1 * np.ones(n - 1))
    x.append(2 * np.ones(n))
    x.append(-1 * np.ones(n - 1))
    npt.assert_array_equal(x[0], np.diag(b, -1))
    npt.assert_array_equal(x[1], np.diag(b, 0))
    npt.assert_array_equal(x[2], np.diag(b, 1))


def test_tridiag_c():
    """Simple test of tridiag. Check with 4 scalar args and known
    eigenvalues"""
    c = 1
    d = 2
    e = 3
    n = 10
    a = rogues.tridiag(n, c, d, e)
    b = a.todense()
    w, v = nl.eig(b)
    w.sort()
    tw = np.array([np.cos(i * np.pi / (n + 1)) for i in range(1, n + 1)])
    tw = d + 2 * np.sqrt(c * e) * tw
    tw.sort()
    res = nl.norm(w - tw)
    npt.assert_almost_equal(res, 0.0, 12)


def test_vand():
    """Simple test of vand."""
    n = 10
    a = np.arange(1, n)
    b = rogues.vand(a)
    c = np.vander(a)
    npt.assert_equal(b, c.T[::-1, :])


def test_vecperm():
    """Simple test of vecperm."""
    n = 5
    a = rogues.vecperm(n)
    npt.assert_equal(nl.det(a), 1)
    npt.assert_equal(a.sum(), n * n)

if __name__ == "__main__":
    npt.run_module_suite()


#IgnoreException
#NumpyTest
#NumpyTestCase
#ParametricTestCase
#TestCase
#Tester
#__builtins__
#__doc__
#__file__
#__name__
#__path__
#assert_almost_equal
#assert_approx_equal
#assert_array_almost_equal
#assert_array_equal
#assert_array_less
#assert_equal
#assert_raises
#assert_string_equal
#build_err_msg
#dec
#decorate_methods
#decorators
#importall
#jiffies
#measure
#memusage
#nosetester
#numpytest
#parametric
#print_assert_equal
#raises
#rand
#restore_path
#run_module_suite
#rundocs
#runstring
#set_local_path
#set_package_path
#test
#utils
#verbose
