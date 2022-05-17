import numpy as np
from scipy.optimize import linprog


def fun_support(mat_poly_set, vec_half_plane_le, b_ub):
    """
    polyhedra set is specified by np.matmul(mat_poly_set, x) <= b_ub=1
    # solver used is in the form:
    c @ x
    such that::
    A_ub @ x <= b_ub
    A_eq @ x == b_eq
    lb <= x <= ub
    """

    # ub:upper bound
    if b_ub is None:
        b_ub = 1.0 * np.ones(mat_poly_set.shape[0])
    # NOTE: b_ub should be consistent with mat_poly_set

    min_neg = linprog(
        -1.0*vec_half_plane_le,  # convert to maximize
        A_ub=mat_poly_set,   # constraint should not revert sign
        # NOTE: not K-step backward reachability constraint matrix!
        b_ub=b_ub)

    max_val = -1.0 * min_neg.fun  # max value of original problem
    return max_val
