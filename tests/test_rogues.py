from scipy.sparse.linalg import bicg
import numpy as np
import numpy.linalg as nl
import numpy.testing as npt
import rogues


def assert_tridiag(a):
    """Return true if input array is tridiagonal"""
    # input arrays must be square
    try:
        if len(a.shape) != 2:
            return False
        m, n = a.shape
        if m != n:
            assert(False)
    except AttributeError:
        assert(False)

    tri = np.tril(np.triu(a, -1), 1)
    npt.assert_array_almost_equal(tri, a)


def test_cauchy():
    """Simple test of cauchy.  We test by checking against a hilbert
    matrix
    """
    x = np.arange(1, 11) - 0.5
    a = rogues.cauchy(x)
    b = rogues.hilb(10)
    npt.assert_array_equal(a, b)

    # scalar argument
    a = rogues.cauchy(10)
    b = rogues.cauchy(np.arange(1, 10))
    npt.assert_array_equal(a, b)

def cheb(xx):
    """Helper function: generator for Chebyshev polynomials"""
    x = xx
    a, b = 1, x
    yield 1.
    while True:
        yield b
        a, b = b, 2. * x * b - a


def test_chebspec():
    """Simple test of chebspec."""
    n = 19
    a = rogues.chebspec(n, k=1)
    w, v = nl.eig(a)
    assert((w.real < 0).all())


def test_chebvand():
    """Simple test of chebvand.  We test by checking against a chebyshev
    polynomial calculated by the test helper function cheb.
    """
    x = 0.8 * np.ones(8)
    a = rogues.chebvand(x)
    b = cheb(0.8)
    c = np.array([next(b) for i in range(8)])
    npt.assert_array_equal(a[:, 0], c)


def test_chow():
    """Simple test of chow.  We test by checking the only the number
    of zero eigenvalues.
    """
    n = 9
    a = rogues.chow(n)
    p = int(np.floor(n / 2))
    num_zero_eigvals = 0
    w, v = nl.eig(a)
    # sad, sad convergence here...
    num_zero_eigvals = (np.abs(w) < 1e-3).sum()
    assert(p == num_zero_eigvals)


def test_clement_a():
    """Simple Test of clement"""
    a = rogues.clement(9)
    w, v = nl.eig(a)
    npt.assert_almost_equal(w.sum(), 0.0, 10)


def test_clement_b():
    """Simple Test of clement, symmetric case"""
    a = rogues.clement(9, k=1)
    w, v = nl.eig(a)
    npt.assert_almost_equal(w.sum(), 0.0, 10)


def test_comp():
    """Simple Test of comp"""
    a  = np.outer(np.arange(1, 7), np.ones(6))
    c  = rogues.comp(a)
    ck = np.array([[1, -1, -1, -1, -1, -1],
                   [-2, 2, -2, -2, -2, -2],
                   [-3, -3, 3, -3, -3, -3],
                   [-4, -4, -4, 4, -4, -4],
                   [-5, -5, -5, -5, 5, -5],
                   [-6, -6, -6, -6, -6, 6]])
    npt.assert_array_equal(c, ck)


def test_compan():
    """Simple Test of compan"""
    a  = np.array([1, 0, -7, 6])
    c  = rogues.compan(a)
    ck = np.array([[0, 7, -6],
                   [1, 0, 0],
                   [0, 1, 0]])
    npt.assert_array_equal(c, ck)


def test_condex():
    """Simple test of condex.  Only tests k=1, should test others"""
    a = np.array([[1, -1, -200, 0],
                  [0, 1, 100, -100],
                  [0, 1, 101, -101],
                  [0, 0, 0, 100]])
    b = rogues.condex(4, 1)
    npt.assert_array_equal(a, b)


def test_cycol():
    """Simple test of cycol(n) for n = 5."""
    a = rogues.cycol(5)
    # Every column should be the same
    result = True
    for i in range(1, 5):
        result = result and (a[:, i] == a[:, 0]).all()
    assert(result)


def test_dingdong():
    """Simple test of dingdong(n) for n = 9. Only tests to
    see if matrix is symmetric. Again, very lame testing,
    but it is a start"""
    a = rogues.dingdong(9)
    # Matrix must be Hermetian
    assert((np.triu(a) == np.triu(np.conj(a.T))).all())


def test_dorr():
    """Simple test of dorr(10).  We actually only check to
    see if the eigenvalue condition numbers are big"""
    a = rogues.dorr(10, r_matrix=True)
    w, v, cond = rogues.condeig(a)
    assert((cond.sum() > 1e15))


def test_dramadah_a():
    """Simple test of dramadah(10,1).  We check to that abs(det()) == 1"""
    a = rogues.dramadah(10)
    npt.assert_almost_equal(abs(nl.det(a)), 1, 12)


def test_dramadah_b():
    """Simple test of dramadah(10, 2).  We check that all eigenvalues == 1"""
    a = rogues.dramadah(10, 2)
    w, v = nl.eig(a)
    npt.assert_array_equal(w, np.ones(10))


def fib():
    """Helper function: generator for Fibonnaci numbers"""
    a, b = 0, 1
    while True:
        yield b
        a, b = b, a + b


def test_dramadah_c():
    """Simple test of dramadah(10, 3).  We check that it's determinant is
    the n-th Fibonacci number"""
    a = rogues.dramadah(10, 3)
    x = nl.det(a)
    gf = fib()
    y = [next(gf) for i in range(10)]
    npt.assert_almost_equal(x, y[9])


def test_fiedler():
    """Simple test of fiedler(n) for n = 3. Taken from Higham &
       Higham p. 135
    """
    a = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    b = rogues.fiedler(3)
    npt.assert_array_equal(a, b)


def test_forsythe():
    """Simple test of forsythe(10, 3.4, .2)"""
    a = rogues.forsythe(10, .1, 9)
    w, v = nl.eig(a)
    npt.assert_almost_equal(np.abs(w.sum()), 90, 12)


def test_frank():
    """Simple exercise of the function frank(n)"""
    a = rogues.frank(5)
    npt.assert_almost_equal(1., np.linalg.det(a), 5)


def test_gearm():
    """Simple test of gearm(10)"""
    a = rogues.gearm(10)
    w, v = nl.eig(a)
    npt.assert_almost_equal(nl.norm(w), 4., 14)


def test_gfpp():
    """Simple exercise of generating the gfpp matrix
    we know this matrix has a growth factor given by 2**(n - 1), which
    for n = 5 is 16, so we just check for that (assumes ge is correct)
    """
    a = rogues.gfpp(5)
    l, u, r = rogues.ge(a)
    npt.assert_almost_equal(r, 16., 8)


def test_grcar():
    """Simple test of grcar"""
    n = 28
    a = rogues.grcar(n)
    w, v = nl.eig(a)
    npt.assert_almost_equal(w.sum(), n, 12)


def test_hadamard():
    """Simple test of hadamard(16)"""
    n = 16
    a = rogues.hadamard(n)
    dm = a.T @ a
    npt.assert_array_equal(np.diag(dm), n * np.ones(n))


def test_hankel():
    """Simple test of Hankel matrix"""
    a = np.arange(10)
    b = np.arange(9, 19)
    h = rogues.hankel(a, b)

    t = np.array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
                  [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
                  [ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11],
                  [ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
                  [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13],
                  [ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14],
                  [ 6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
                  [ 7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
                  [ 8,  9, 10, 11, 12, 13, 14, 15, 16, 17],
                  [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]])

    npt.assert_array_equal(h, t)


def test_hanowa():
    """Simple test of hanowa matrices. Real parts of eigenvalue
       must all be one
    """
    n = 10
    a = rogues.hanowa(n)
    w, v = nl.eig(a)
    npt.assert_array_equal(w.real, np.ones(n))


def test_hilb():
    """Simple test of hilb() Hilbert matrices"""
    a = rogues.hilb(10)
    cond = nl.cond(a)
    assert(cond > 1e+12)


def test_invhess():
    """
    Simple test of invhess matrix
    This is really not a very good test, for I don't know of
    any special properties to look for.  However, it looks to
    be very poorly conditioned, so we will just look for that.
    """
    a = rogues.invhess(55)
    w, v, cond = rogues.condeig(a)
    assert((cond > 1e+14).any())


def test_invol():
    """Simple test of invol.  Must be it's own inverse."""
    n = 5
    a = rogues.invol(n)
    b = a @ a
    bd = np.diag(np.ones(n))
    print ("invol**2 = ", b)
    npt.assert_array_almost_equal(b, bd)


def test_ipjfact():
    """Simple test of ipjfact. Use simple minded version"""
    fa = np.arange(200.)
    fa[0] = 1.
    fac = np.cumprod(fa)
    n = 6
    a = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            a[i, j] = fac[i + j + 2]
    b = nl.det(a)
    c, d = rogues.ipjfact(n)
    assert((a == c).all() and ((np.abs(b - d) / np.abs(b)) < 1e-10))


def test_jordbloc():
    """Simple test of jordbloc. Do we recover the correct eigenvalues?"""
    evalue = 5.911  # arbitrary value
    a = rogues.jordbloc(10, evalue)
    w, v = nl.eig(a)
    b = evalue * np.ones(10)
    npt.assert_array_equal(w, b)


def test_kahan():
    """Simple test of kahan matrices.
       In fact, it is very cheesey.  I just check against diag element
       generated by NIST's matrixmarket deli for the kahan matrix
    """
    nist_diag = np.array([1.0000000000000555, \
                          0.9320390859672762, \
                          0.8686968577706671, \
                          0.8096594252991716, \
                          0.7546342307005867, \
                          0.7033485986217525, \
                          0.6555483849757441, \
                          0.6109967175400799, \
                          0.5694728221450271, \
                          0.5307709286352231])
    a = rogues.kahan(10)
    print (np.diag(a))
    print (nist_diag)
    npt.assert_array_almost_equal(np.diag(a), nist_diag)


def test_kms():
    """Test of kms matrix. Tests against a known L*D*L.T factorization"""
    n = 4
    rho = 0.5
    a = rogues.kms(n, rho)

    #  The following is an advertised property of the kms matrix, but I
    #  have not been able to verify.  The kms matrices are very simple, and
    #  easy to verify by inspection so the problem is either in the stated
    #  property or in my implementation of it.
    l = nl.inv(rogues.triw(n, -rho, 1).T)
    d = (1 - np.abs(rho) ** 2) * np.eye(n)
    d[0, 0] = 1
    k = l @ d @ l.T
    npt.assert_array_equal(a, k)

    # verify that nl.inv(a) is tridiagonal.
    b = nl.inv(a)
    assert_tridiag(b)


def test_krylov():
    """Simple test of krylov matrix"""
    d = np.array([[1, 10, 40, 160],
                  [2, 10, 40, 160],
                  [3, 10, 40, 160],
                  [4, 10, 40, 160]])
    a = np.ones((4, 4))
    b = np.arange(1, 5)
    c = rogues.krylov(a, b)
    npt.assert_array_equal(c, d)


def test_lauchli():
    """Simple test of the lauchli matrix."""
    a = np.array([[1, 1, 1, 1], [5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0],\
                  [0, 0, 0, 5]])
    b = rogues.lauchli(4, 5)
    npt.assert_array_equal(a, b)


def test_lehmer():
    """Simple test of lehmer matrix."""
    a = np.array([[1 / 1., 1 / 2., 1 / 3., 1 / 4.],
                  [1 / 2., 2 / 2., 2 / 3., 2 / 4.],
                  [1 / 3., 2 / 3., 3 / 3., 3 / 4.],
                  [1 / 4., 2 / 4., 3 / 4., 4 / 4.]])
    b = rogues.lehmer(4)
    npt.assert_array_equal(a, b)


def test_lesp():
    """Simple test of lesp matrix"""
    n = 5
    b = rogues.lesp(n)
    c = np.diag(b, -1)
    d = 1 / np.arange(2., n + 1)
    assert_tridiag(b)
    npt.assert_array_equal(c, d)


def test_lotkin():
    """Simple test of lotkin matrix."""
    a = rogues.lotkin(10)
    # supposed to have many negative eigenvalues of small magnitude
    w, v = nl.eig(a)
    assert((w < 0).sum() == 9  and  (abs(w) < 1e-5).sum() == 5)


def test_minij():
    """Simple test of minij matrix."""
    n = 10
    a = rogues.minij(n)
    b = nl.inv(a)
    assert_tridiag(b)
    assert(b[n - 1, n - 1] == 1)


def test_moler():
    """Simple test of moler(n) for n = 3. Taken from Higham & Higham p. 135"""
    a = np.array([[1, -1, -1], [-1, 2, 0], [-1, 0, 3]])
    b = rogues.moler(3)
    npt.assert_array_equal(a, b)


def test_neumann():
    """Simple test of neumann matrix."""
    # OK, this is insane, but these _seem_ to be the eigenvalues...
    ans = np.array([0., 1, 1, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 7, 8])
    a, t = rogues.neumann(16)
    w, v = nl.eig(a)
    x = w.real
    #
    # for some reason, cannot put the x.sort() directly into the allclose func
    # and must do it first...
    x.sort()
    assert(np.allclose(x, ans))


def test_ohess():
    """Simple test of ohess matrix."""
    n = 10
    a = rogues.ohess(n)
    # Test to see if a is orthogonal...
    b = a @ a.T
    assert(np.allclose(b, np.eye(n)))

# def test_othog():
#     """Simple test of orthog matrix."""
#     # We actually need to test all the options, but...
#     n = 10
#     a = rogues.orthog(n)
#     # Test to see if a is orthogonal...
#     b = a @ a.T
#     assert( np.allclose(b, np.eye(n)) )


def test_parter():
    """Simple test of parter matrix."""
    # Well, this is not a real test, only a twiddle
    n = 10
    a = rogues.parter(n)
    npt.assert_almost_equal(np.abs(nl.det(a)), 33961.9329448)


def test_pascal_0():
    """Simple test of pascal matrix: k = 0."""
    # Note that this test will fail if n > 12
    n = 12
    a = rogues.pascal(n)
    w, v = nl.eig(a)
    b = w * w[::-1]
    npt.assert_array_almost_equal(b, np.ones(n))


def test_pascal_1():
    """Simple test of pascal matrix: k = 1."""
    # Notice we recover the unit matrix with n = 18, better than previous test
    n = 18
    a = rogues.pascal(n, 1)
    b = a @ a
    assert(np.allclose(b, np.eye(n)))


def test_pdtoep():
    """Simple test of pdtoep matrix."""
    # just test positive definite at some random vector
    n = 10
    a = rogues.pdtoep(n)
    x = np.random.rand(n)
    b = np.dot(x, np.dot(x, a))
    assert(b > 0)


def test_pei():
    """Simple test of pei matrix."""
    n = 10
    a = rogues.pei(n)
    npt.assert_almost_equal(nl.det(a), 11.0)


def test_pentoep():
    """Simple test of pentoep matrix."""
    # incredibly lame, essentially just test that we don't assert
    n = 18
    a = rogues.pentoep(n)
    assert(np.abs(nl.det(a)) > 1.0)


def test_poisson():
    """Simple test of poisson matrix."""
    # incredibly lame, essentially just test that we don't assert
    n = 10
    a = rogues.poisson(n)
    assert(a.shape[0] == 100)


def test_prolate():
    """Simple test of prolate matrix."""
    n = 18
    a = rogues.prolate(n)
    w, v = nl.eig(a)
    assert((w > 0).all() and (w < 1).all())


def test_qmult():
    """Simple test of qmult."""
    n = 18
    a = rogues.qmult(n)
    npt.assert_almost_equal(np.abs(nl.det(a)), 1.0)


def test_rando():
    """Simple test of rando matrix generation."""
    n = 18
    a = rogues.rando(n, k=2)
    assert((np.abs(a)).sum() == n ** 2)


def test_randsvd():
    """Simple test of randsvd THIS WILL FAIL ~10% of the time"""
    n = 18
    k = 1e+20
    a = rogues.randsvd(n, kappa=k)
    c = nl.cond(a) / k   # so c _should_ be close to one (yeah, right)

    # The actual condition number should be within an order of magnitude
    # of what we asked for but, sometimes even a order of magnitude doesn't
    # quite cut it so we loosen up even more (even this doesn't always pass
    # so we must think of a better check (looks like this fails a little
    # less than 10% of the time.
    #print ('c = %f' % c)
    #
    assert(c > 0.001 and c < 1000.)


def test_redheff():
    """Simple test of Redheffer matrix."""
    n = 20
    num_one_eigs = n - np.floor(np.log2(n)) - 1
    a = rogues.redheff(n)
    w, v = nl.eig(a)
    count = 0
    for x in w:
        if round(x.real, 3) == 1:
            count += 1

    npt.assert_equal(num_one_eigs, count)


def test_riemann():
    """Simple test of the Riemann matrix."""
    n = 10
    a = rogues.riemann(n)
    b = np.tril(-np.ones((n, n)), -1)
    c = np.tril(a, -1)
    # Kind of a goofy prop to check, but it's simple
    npt.assert_array_equal(b, c)


def test_smoke():
    """Simple test of the smoke matrix."""
    n = 32
    a = rogues.smoke(n)
    w, v = nl.eig(a)
    assert(
        np.allclose(np.sqrt((w * np.conj(w)).real), 2 ** \
                              (1 / float(n)) * np.ones(32)))


def test_triw():
    """Simple test of triw(n) for n = 5"""
    a = np.array([[1, -1, -1, -1, -1],
                  [0, 1, -1, -1, -1],
                  [0, 0, 1, -1, -1],
                  [0, 0, 0, 1, -1],
                  [0, 0, 0, 0, 1]])
    b = rogues.triw(5)
    npt.assert_array_equal(a, b)


def test_wathen():
    """Simple test of wathen(n,n) for n = 5"""
    a = rogues.wathen(5, 5)
    n = a.shape[0]
    b = np.ones(n)
    x, info = bicg(a, b)
    npt.assert_equal(info, 0)
    # since wathen is symmetric, positive definite,
    # it had better have all real eigs
    npt.assert_almost_equal((x.imag).sum(), 0.0)


def test_wilk():
    """Simple test of wilk(4)"""
    a, b = rogues.wilk(4)
    w = np. array([[0.9143e-4, 0, 0, 0],
                   [0.8762, 0.7156e-4, 0, 0],
                   [0.7943, 0.8143, 0.9504e-4, 0],
                   [0.8017, 0.6123, 0.7165, 0.7123e-4]])
    npt.assert_array_equal(a, w)


def test_wilkinson():
    """Simple test of wilkinson(21)"""
    a, b = rogues.wilk(21)
    w = rogues.wilkinson(21)
    npt.assert_array_equal(a, w)


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
