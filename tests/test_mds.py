import numpy as np
import numpy.testing as npt
import rogues


def rosen(x, *varargs):
    """
    Here we actually compute the negative of the Rosenbrock function
    since mdsmax attempts to find the maximum.
    """
    sum = 0.
    for i in range(x.size - 1):
        sum += (1 - x[i]) ** 2 + 100 * ((x[i + 1] - x[i] ** 2) ** 2)

    return -sum


def exp_func(x, *varargs):
    """
    Very simple function to maximize.  sum over i of exp(-x[i]**2)
    """
    sum = 0.
    for z in x:
        sum += np.exp(-z ** 2)

    return sum


def test_mds_exp():
    n = 5
    z = np.ones(n)
    x, fmax, nf = rogues.mdsmax(exp_func, z)
    va = []
    npt.assert_almost_equal(exp_func(np.zeros(n), va), exp_func(x, va))


def test_mds_rosen():
    # so the starting point is kind of cooked...
    n = 5
    z = 0.9 * np.ones(n)
    stopit = (1e-12, np.inf, np.inf, 0, 1)
    x, fmax, nf = rogues.mdsmax(rosen, z, stopit)
    npt.assert_almost_equal(np.ones(n), x)


if __name__ == "__main__":
    npt.run_module_suite()
