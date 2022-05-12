import numpy as np
from scipy.optimize import linprog

def demo_linprog():
    """demo_linprog."""

    help(linprog)
    A = np.matrix([
        [1, 1],
        [1, 0.25],
        [1, -1],
        [-0.25, -1],
        [-1, -1],
        [-1, 1],
    ])
    b = np.matrix([2, 1, 2, 1, -1, 2])
    c = np.array([-1, -1/3])
    res = linprog(c, A_ub=A, b_ub=b)
    dir(res)
    res.fun
