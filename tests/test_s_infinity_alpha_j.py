import numpy as np
from tmpc.mpc_qp_tube import MPCqpTube
from tmpc.constraint_x_u_couple import ConstraintStageXU
from tmpc.utils_case import Probset
from tmpc.constraint_s_inf import ConstraintSAlpha
from tmpc.support_set_inclusion import is_implicit_subset_explicit
import control


def test_set_inclusion():
    """
    # For two set
    Y={y=Ax|Mx<=1}  # in implicit form
    Z={x|Nx<=1}  # in explicit form
    to verify:  Y \\subset Z
    <=> {max}_x{N*A*x}=max_x{Ny|y=Ax, Mx<=1} <=1
    s.t. Mx<=1  (set {x|Mx<=1} defines Y={Ax})
    <=> if the maximum value of N*y: y=Ax & Mx<=1
    is smaller than 1, then y is inside Z.
    <=> h({Mx<=1}, N*A} <= 1
    note
    A.shape = [dim(x), dim(x)]
    M.shape = [m, dim(x)]
    N.shape = [n, dim(x)]
    (N*A).shape = [n, dim(x)]
    """
    #is_implicit_subset_explicit()
    pass

def pole_place():
    mat_sys = [[-1, 1],  [0, 1]]
    mat_sys = np.array(mat_sys)
    mat_input = [[0], [1]]
    mat_input = np.array(mat_input)
    mat_k = control.place(mat_sys, mat_input, [-2, -5])
    mat_c = mat_sys - mat_input* mat_k
    np.linalg.eig(mat_c)

def test_j_alpha():
    prob = Probset()
    mat_constraint4w = np.array([[1, 0], [0, 1]])
    prob.mat_sys
    np.linalg.eig(prob.mat_sys)
    mat_k_s = prob.mat_k * (10)  # why not ?
    mat_k_s = np.array([[0, 1]]) * (0.01) # why not
    mat_k_s = control.place(prob.mat_sys, prob.mat_input, [-0.2, -0.1])
    mat_k_s = -1.0 * mat_k_s
    np.matmul(prob.mat_input, mat_k_s)
    mat_c = prob.mat_sys + np.matmul(prob.mat_input, mat_k_s)
    np.linalg.eig(mat_c)
    constraint_j_alpha = ConstraintSAlpha(
        mat_sys=prob.mat_sys,
        mat_input=prob.mat_input,
        mat_k_s=mat_k_s,
        mat_w=mat_constraint4w,
        max_iter=100)
    constraint_j_alpha.cal_power_given_alpha(0.9)
    constraint_j_alpha.cal_power_given_alpha(0.1)
    constraint_j_alpha.cal_power_given_alpha(0.01)
    constraint_j_alpha.cal_power_given_alpha(0.001)
    # constraint_j_alpha.cal_alpha_given_power(3)
