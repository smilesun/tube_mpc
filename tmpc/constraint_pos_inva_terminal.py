import numpy as np
from tmpc.utils_plot_constraint import plot_polytope
from tmpc.support_set_inclusion import fun_is_subset, is_set_in_half_plane


class PosInvaTerminalSetBuilder():
    """PosInvaTerminalSetBuilder."""

    def __init__(self, mat_sys,
                 mat_state_constraint,
                 tolerance
                 ):
        """__init__.
        :param mat_sys: closed loop dynamic A_c=A+BK
        :param mat_state_constraint: Mx<=1
        :param tolerance: to udge backward inclusion,
        the linear program might return a value close
        to 1.
        """
        self.tolerance = tolerance
        self.mat_state_constraint = mat_state_constraint
        self.mat_sys = mat_sys

    def __call__(self, max_iter):
        """__call__.
        :param max_iter: maximum number of iterations
        """
        mat_reach_constraint = iterate_invariance(
            mat0=self.mat_state_constraint,
            A=self.mat_sys,
            max_iter=max_iter,
            tolerance=self.tolerance)
        return mat_reach_constraint


def augment_mat_k(mat_sys, mat_k, mat0,
                  tolerance,
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
        if not is_set_in_half_plane(mat_kp1, row, tolerance):
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
                       max_iter, verbose=True, call_back=None):
    """iterate_invariance.
    :param mat0:
    :param A:
    :param max_iter: maximum number of iterations
    :param verbose:
    :param call_back:
    """
    mat_k_backstep = mat0
    counter = 0
    for k in range(max_iter):
        mat_k_old = mat_k_backstep
        mat_k_backstep = augment_mat_k(mat_k=mat_k_backstep,
                                       mat0=mat0,
                                       mat_sys=A,
                                       tolerance=tolerance,
                                       call_back=call_back)
        if verbose:
            print("iteration %d" % (k))
            print(mat_k_backstep)
        if call_back:
            call_back(mat_k_backstep, "iteration %d" % (k))

        # backward reachability S_k \subset S_{k-1} is always satisfied
        # since it is more difficult for backward k steps to satisfies
        # constraint and each step must satisfy the common constraint
        # if S_{k-1} \subset S_{k}, then iteration can stop

        if fun_is_subset(mat_k_old,
                         mat_k_backstep,
                         tolerance):
            print("inverse inclusion detected")
            print(mat_k_backstep, mat_k_old)
            break
        counter += 1
    if counter >= max_iter:
        raise RuntimeError("positive invariant set not found with in maximum\
                           number of iterations!")
    return mat_k_backstep


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
    constraint = iterate_invariance(mat0=M0, A=A, max_iter=22,
                                    tolerance=0)
    constraint = iterate_invariance(mat0=M0, A=A, max_iter=30,
                                    tolerance=1e-4)
    plot_polytope(constraint)
    plot_polytope(M0)
    constraint = iterate_invariance(mat0=M0, A=A, max_iter=3,
                                    tolerance=0,
                                    call_back=plot_polytope)
