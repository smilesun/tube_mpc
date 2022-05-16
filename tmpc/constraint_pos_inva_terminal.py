import numpy as np
import matplotlib.pyplot as plt
from control import matlab
from scipy.optimize import linprog
from tmpc.utils_plot_constraint import plot_polytope


class PosInvaTerminalSetBuilder():
    """PosInvaTerminalSetBuilder."""

    def __init__(self, mat_sys,
                 mat_state_constraint,
                 tolerance=0.01
                 ):
        """__init__.

        :param mat_sys:
        :param mat_state_constraint:
        """
        self.tolerance = tolerance
        self.mat_state_constraint = mat_state_constraint
        self.mat_sys = mat_sys

    def __call__(self, n_iter):
        """__call__.

        :param n_iter:
        """
        mat_reach_constraint = iterate_invariance(
            mat0=self.mat_state_constraint,
            A=self.mat_sys,
            n_iter=n_iter,
            tolerance=self.tolerance)
        return mat_reach_constraint


def is_set_in_half_plane(mat_poly_set, half_plane_le,
                         b_ub=None,
                         tolerance=0,
                         fun_criteria=lambda x: x < 1):
    """
    polyhedra set is specified by np.matmul(mat_poly_set, x) <= 1
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
        -1.0*half_plane_le,  # convert to maximize
        A_ub=mat_poly_set,   # constraint should not revert sign
        # NOTE: not K-step backward reachability constraint matrix!
        b_ub=b_ub)

    max_val = -1.0 * min_neg.fun  # max value of original problem

    if fun_criteria(max_val+tolerance):
        # if worst case satisfies constraint, then no need for this constraint
        # to exist
        return True
    return False


def augment_mat_k(mat_sys, mat_k, mat0,
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
        row = mat_candidate[i, ]
        if not is_set_in_half_plane(mat_kp1, row):
            mat_kp1 = np.vstack((mat_kp1, row))
            # FIXME: not stack M0!
            if call_back:
                call_back(mat_kp1, "the %d th row: %s" % (i, str(row)))
    return mat_kp1  # FIXME:  return should be outside for loop!


def test_augment_mat_k():
    """test_augment_mat_k."""
    A = np.matrix([[1.1, 0.5], [0.5, 0.9]])
    np.linalg.eig(A)
    M0 = np.matrix(
        [[0, 1],
         [1, 0],
         [-1, 0],
         [0, -1]])
    augment_mat_k(mat_k=M0, mat0=M0, mat_sys=A)


def iterate_invariance(mat0, A,
                       tolerance,
                       n_iter=10, verbose=True, call_back=None):
    """iterate_invariance.
    :param mat0:
    :param A:
    :param n_iter: maximum number of iterations
    :param verbose:
    :param call_back:
    """
    mat_k_backstep = mat0
    for k in range(n_iter):
        mat_k_old = mat_k_backstep
        mat_k_backstep = augment_mat_k(mat_k=mat_k_backstep,
                                       mat0=mat0,
                                       mat_sys=A,
                                       call_back=call_back)
        if verbose:
            print("iteration %d" % (k))
            print(mat_k_backstep)
        if call_back:
            call_back(mat_k_backstep, "iteration %d" % (k))
        if fun_is_set_include(mat_k_backstep,
                              mat_k_old,
                              tolerance):
            print("inverse inclusion detected")
            print(mat_k_backstep, mat_k_old)
            break
    return mat_k_backstep


def fun_is_set_include(mat_set1, mat_set2, tolerance):
    """
    mat_set1 \\in mat_set2
    """
    for i in range(mat_set2.shape[0]):
        half_plane_le = mat_set2[i]
        if not is_set_in_half_plane(mat_set1, half_plane_le,
                                    tolerance=tolerance):
            return False
    return True


def test_iterate_invariance():
    """test_iterate_invariance."""
    #  A = np.matrix([[0.3, 0.5], [0.5, 0.3]])
    #  # A is stable, so one step reach invariant
    #  initial set is maximum
    A = np.matrix([[1.1, 0.5], [0.5, 0.9]])
    np.linalg.eig(A)
    M0 = np.matrix(
        [[0, 1],
         [1, 0],
         [-1, 0],
         [0, -1]])
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=20,
                                    tolerance=0)
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=30,
                                    tolerance=1e-4)
    plot_polytope(constraint)
    plot_polytope(M0)
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=3,
                                    tolerance=0,
                                    call_back=plot_polytope)
