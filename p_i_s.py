import numpy as np
import matplotlib.pyplot as plt
from control import matlab
from scipy.optimize import linprog
from utils_plot_constraint import plot_polytope


def augment_mat_k(mat_sys, mat_k, mat0,
                  fun_criteria=lambda x: x < 1):
    """augment.
    t_N feasibility: M0
    1-step backward feasibility: M_1: [M_0A] and M_0
    2-step backward feasibility: M_2: [M_1A] and M_0
    :param mat0:  stage constraint
    :param mat_k: initial value M_0, return of this function as M_{k+1}
    for next iteration
    :param mat_sys: discrete time system dynamic
    :param criteria:
    """
    mat_kp1 = mat0
    mat_candidate = np.matmul(mat_k, mat_sys)   # M_k*(Ax)<=1
    nrow = mat_candidate.shape[0]
    for i in range(nrow):
        row = -1 * mat_candidate[i, ]  # FIXME: default linprog minimize, here maximize needed
        # ub:upper bound
        b_ub = 1.0 * np.ones(mat_kp1.shape[0])  # FIXME: this line should be inside for loop, mat_kp1 not mat_k!
        res = linprog(row,
                      A_ub=mat_kp1,  # FIXME: not mat_k!
                      b_ub=b_ub)
        max_val = res.fun
        if fun_criteria(max_val):
            mat_kp1 = np.vstack((mat0, row))
    return mat_kp1 # FIXME:  return should be outside for loop!

def test_augment_mat_k():
    from script_input import M0, A
    augment_mat_k(M0, M0, A)


def iterate_invariance(mat0, A, n_iter=10, verbose=True, call_back=None):
    mat_k = mat0
    for _ in range(n_iter):
        mat_k = augment_mat_k(mat_k=mat_k, mat0=mat0, mat_sys=A)
        if verbose:
            print(mat_k)
        if call_back:
            call_back(mat_k)
    return mat_k

def test_iterate_invariance():
    from script_input import M0, A
    plot_polytope(M0)
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=10, call_back=plot_polytope)
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=100)
