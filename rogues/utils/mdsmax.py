import numpy as np
import numpy.linalg as nl

status_file = None
trace = False


def print_status(msg):
    if trace and status_file is not None:
        status_file.write(msg)


def mdsmax(fun, x, stopit=None, savit=None, varargin=[]):
    """
    MDSMAX  Multidirectional search method for direct search optimization.
        [x, fmax, nf] = MDSMAX(FUN, x0, STOPIT, SAVIT) attempts to
        maximize the function FUN, using the starting vector x0.
        The method of multidirectional search is used.
        Output arguments:
               x    = vector yielding largest function value found,
               fmax = function value at x,
               nf   = number of function evaluations.
        The iteration is terminated when either
               - the relative size of the simplex is <= stopit[0]
                 (default 1e-3),
               - stopit[1] function evaluations have been performed
                 (default inf, i.e., no limit), or
               - a function value equals or exceeds stopit[2]
                 (default inf, i.e., no test on function values).
        The form of the initial simplex is determined by STOPIT[3]:
          stopit[3] = 0: regular simplex (sides of equal length, the default),
          stopit[3] = 1: right-angled simplex.
        Progress of the iteration is not shown if stopit[4] = 0 (default 1).
        If a non-empty fourth parameter string SAVIT is present, then
        `SAVE SAVIT x fmax nf' is executed after each inner iteration.
        NB: x0 can be a matrix.  In the output argument, in SAVIT saves,
            and in function calls, x has the same shape as x0.
        MDSMAX(fun, x0, STOPIT, SAVIT, P1, P2,...) allows additional
        arguments to be passed to fun, via feval(fun,x,P1,P2,...).

    This implementation uses 2n^2 elements of storage (two simplices), where x0
    is an n-vector.  It is based on the algorithm statement in [2, sec.3],
    modified so as to halve the storage (with a slight loss in readability).

    References:
    [1] V. J. Torczon, Multi-directional search: A direct search algorithm for
        parallel machines, Ph.D. Thesis, Rice University, Houston, Texas, 1989.
    [2] V. J. Torczon, On the convergence of the multidirectional search
        algorithm, SIAM J. Optimization, 1 (1991), pp. 123-145.
    [3] N. J. Higham, Optimization by direct search in matrix computations,
        SIAM J. Matrix Anal. Appl, 14(2): 317-333, 1993.
    [4] N. J. Higham, Accuracy and Stability of Numerical Algorithms,
        Second edition, Society for Industrial and Applied Mathematics,
        Philadelphia, PA, 2002; sec. 20.5.
    [5] T.G. Kolda, R. M. Lewis, V. Torczon, Optimization by Direct Search: New
        Perspectives on Some Classical and Modern Methods", SIAM Review,
        Vol. 45, No. 3, pp. 385-482
    """
    global status_file
    global trace

    # Stopping with 1e-3 actually leads to wrong results..
    #
    if stopit is None:
        stopit = (1e-5, np.inf, np.inf, 0, 1)

    x0 = x
    n = x0.size

    varargin = []

    mu = 2             # Expansion factor.
    theta = 0.5        # Contraction factor.

    # Set up convergence parameters etc.
    # Tolerance for cgce test based on relative size of simplex.
    tol = stopit[0]
    trace = stopit[4]

    if savit is not None:
        status_file = open(savit, 'w')

    v = np.hstack((np.zeros((n, 1)), np.eye(n)))
    t = v
    f = np.zeros(n + 1)
    ft = f
    v[:, 0] = x0
    f[0] = fun(x, varargin)
    fmax_old = f[0]

    print_status('f(x0) = %9.4e\n' % f[0])

    # Set up initial simplex.
    scale = max(nl.norm(x0, np.inf), 1)
    if stopit[3] == 0:
        # Regular simplex - all edges have same length
        # Generated from construction given in reference [18, pp. 80-81] of [1]
        alpha = scale / (n * np.sqrt(2)) * np.hstack((np.sqrt(n + 1) - 1 + n,
                                                     np.sqrt(n + 1) - 1))
        v[:, 1:n + 1] = np.outer(x0 + alpha[1] * np.ones(n), np.ones(n))
        for j in range(1, n + 1):
            v[j - 1, j] = x0[j - 1] + alpha[0]
            x = v[:, j]
            f[j] = fun(x, varargin)

    else:
        # Right-angled simplex based on co-ordinate axes.
        alpha = scale * np.ones(n + 1)
        for j in range(1, n + 1):
            v[:, j] = x0 + alpha[j] * v[:, j]
            x = v[:, j]
            f[j] = fun(x, varargin)

    nf = n + 1
    size = 0           # Integer that keeps track of expansions/contractions.
    flag_break = 0     # Flag which becomes true when ready to quit outer loop.

    k = 0
    m = 0

    while True:
        k = k + 1

        # Find a new best vertex  x  and function value  fmax = f(x).
        fmax = np.max(f)
        j = np.argmax(f)
        # swap_columns 0, j
        v[:, [0, j]] = v[:, [j, 0]]
        v0 = v[:, 0]
        if savit is not None:
            x = v0
            print_status(' x: %2.6e  fmax: %2.6e  nf: %d\n' % (x, fmax, nf))

        f[0], f[j] = f[j], f[0]

        print_status('Iter. %2.0f,  inner = %2.0f,  size = %2.0f,  ' %
                     (k, m, size))

        print_status('nf = %3.0f,  f = %9.4e  (%2.1f)\n' %
                     (nf, fmax, 100 * (fmax - fmax_old) /
                      (abs(fmax_old) + np.finfo(float).eps)))

        fmax_old = fmax

        # Stopping Test 1 - f reached target value?
        if fmax >= stopit[2]:
            msg = 'Exceeded target...quitting\n'
            break

        m = 0

        while True:
            m = m + 1

            # Stopping Test 2 - too many f-evals?
            if nf >= stopit[1]:
                msg = 'Max number of function evaluations exceeded...quitting'
                flag_break = 1
                break

            # Stopping Test 3 - converged?   This is test (4.3) in [1].
            vt = np.outer(v0, np.ones(n))
            # size_simplex = nl.norm(v[:,1:n+1] - v0[:,np.ones(n)], 1) /  \
            size_simplex = (nl.norm(v[:, 1:n + 1] - vt, 1) /
                            max(1, nl.norm(v0, 1)))
            if size_simplex <= tol:
                msg = 'Simplex size %9.4e <= %9.4e...quitting\n' % \
                               (size_simplex, tol)
                flag_break = 1
                break

            for j in range(1, n + 1):      # ---Rotation (reflection) step.
                t[:, j] = 2 * v0 - v[:, j]
                x = t[:, j]
                ft[j] = fun(x, varargin)

            nf = nf + n

            replaced = (np.max(ft[1:n + 1]) > fmax)

            if replaced:
                for j in range(1, n + 1):   # ---Expansion step.
                    v[:, j] = (1 - mu) * v0 + mu * t[:, j]
                    x = v[:, j]
                    f[j] = fun(x, varargin)

                nf = nf + n
                # Accept expansion or rotation?
                if np.max(ft[1:n + 1]) > np.max(f[1:n + 1]):
                    # Accept rotation.
                    v[:, 1:n + 1] = t[:, 1:n + 1]
                    f[1:n + 1] = ft[1:n + 1]
                else:
                    size += 1    # Accept expansion (f and V already set).

            else:
                for j in range(1, n + 1):      # ---Contraction step.
                    v[:, j] = (1 + theta) * v0 - theta * t[:, j]
                    x = v[:, j]
                    f[j] = fun(x, varargin)

                nf = nf + n
                replaced = (np.max(f[1:n + 1]) > fmax)
                # Accept contraction (f and V already set).
                size -= 1

            if replaced:
                break

            if m % 10 == 0:
                print_status('        ...inner = %2.0f...\n' % m)

        if flag_break:
            break

    # Finished.
    print_status(msg)
    x = v0

    return x, fmax, nf
