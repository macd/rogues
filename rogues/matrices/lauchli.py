import numpy as np


def lauchli(n, mu=None):
    """
    LAUCHLI   Lauchli matrix - rectangular.
          lauchli(n, mu) is the (n+1)-by-n matrix vstack(ones(1), mu*eye(n))
          it is a well-known example in least squares and other problems
          that indicates the dangers of forming a'*a.
          mu defaults to sqrt(eps).

          Reference:
          P. Lauchli, Jordan-Elimination und Ausgleichung nach
          kleinsten Quadraten, Numer. Math, 3 (1961), pp. 226-240.
    """
    if mu is None:
        mu = np.sqrt(np.finfo(float).eps)

    a = np.vstack((np.ones(n), mu * np.eye(n)))

    return a
