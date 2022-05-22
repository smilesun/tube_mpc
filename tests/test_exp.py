import numpy as np
import control
from tmpc.mpc_qp_tube import MPCqpTube
from tmpc.constraint_x_u_couple import ConstraintStageXU
from tmpc.utils_case import Probset
from tmpc.simulate import Exp
from tmpc.dyn_sys import DynSysL
from tmpc.constraint_s_inf import ConstraintSAlpha


def test_exp():
    prob = Probset()
    mat_sys = prob.mat_sys
    mat_input = prob.mat_input
    mat_x = prob.x_only_constraint
    mat_u = prob.u_only_constraint
    mat_w = prob.mat_w

    constraint_x_u = ConstraintStageXU(
        dim_sys=prob.dim_sys,
        dim_input=prob.dim_input,
        mat_x=mat_x,
        mat_u=mat_u)

    mat_k_s = control.place(prob.mat_sys, prob.mat_input, [-0.2, 0.1])
    mat_k_s = -1.0 * mat_k_s

    constraint_j_alpha = ConstraintSAlpha(
        mat_sys=prob.mat_sys,
        mat_input=prob.mat_input,
        mat_k_s=mat_k_s,
        mat_w=prob.mat_w,
        max_iter=100)
    alpha_ini = 0.001
    j_alpha = constraint_j_alpha.cal_power_given_alpha(alpha_ini)

    mpctube = MPCqpTube(
        mat_sys, mat_input,
        mat_q=prob.mat_q,
        mat_r=prob.mat_r,
        mat_k_s=mat_k_s,
        mat_k_z=mat_k_s,
        mat_constraint4w=prob.mat_w,
        constraint_x_u=constraint_x_u,
        j_alpha=j_alpha,
        alpha_ini=alpha_ini,
        tolerance=0.01)
    horizon = 3

    mpctube.build_mat_block_ub(horizon=horizon, j_alpha=j_alpha)
    assert mpctube.mat_ub_block.shape[1] == \
        horizon*(prob.dim_input + prob.dim_sys) + prob.dim_sys \
        + j_alpha * prob.dim_sys

    x = np.array([[0.01, 0.01]]).T
    dyn = DynSysL(dim_sys=prob.dim_sys,
                  dim_u=prob.dim_input,
                  x_ini=x,
                  constraint_x_u=constraint_x_u,
                  max_w=prob.max_w)
    exp = Exp(dyn, controller=mpctube)
    exp.run(200, 3, j_alpha=j_alpha)
