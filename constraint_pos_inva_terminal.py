import numpy as np
import matplotlib.pyplot as plt
from control import matlab
from scipy.optimize import linprog
from utils_plot_constraint import plot_polytope


class PosInvaTerminalSetBuilder():
    def __init__(self, mat_sys,
                 mat_state_constraint):
        self.mat_state_constraint = mat_state_constraint
        self.mat_sys = mat_sys

    def __call__(self, n_iter):
        mat_reach_constraint = iterate_invariance(
            mat0=self.mat_state_constraint,
            A=self.mat_sys,
            n_iter=n_iter)
        return mat_reach_constraint


def augment_mat_k(mat_sys, mat_k, mat0,
                  fun_criteria=lambda x: x < 1,
                  call_back=None):
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
    mat_kp1 = mat0  # FIXME: always start with M0 for testing
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
        if fun_criteria(max_val):  # if worst case satisfies constraint, then no need for this constraint to exist
            mat_kp1 = np.vstack((mat_kp1, row))   # FIXME: not stack M0!
            if call_back:
                call_back(mat_kp1, "the %d th row: %s" % (i, str(row)))
    return mat_kp1 # FIXME:  return should be outside for loop!

def test_augment_mat_k():
    from script_input import M0, A
    augment_mat_k(M0, M0, A)


def iterate_invariance(mat0, A, n_iter=10, verbose=True, call_back=None):
    mat_k_backstep = mat0
    for k in range(n_iter):
        mat_k_backstep = augment_mat_k(mat_k=mat_k_backstep, mat0=mat0, mat_sys=A, call_back=call_back)
        if verbose:
            print("iteration %d" % (k))
            print(mat_k_backstep)
        if call_back:
            call_back(mat_k_backstep, "iteration %d" % (k))
    return mat_k_backstep

def test_iterate_invariance():
    from script_input import M0, A
    plot_polytope(M0)
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=3, call_back=plot_polytope)
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=100)
    plot_polytope(constraint)
