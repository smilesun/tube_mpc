import numpy as np
from tmpc.mpc_qp_tube import MPCqpTube
from tmpc.constraint_x_u_couple import ConstraintStageXU
from tmpc.utils_case import Probset


def test_tube():
    prob = Probset()
    mat_sys = prob.mat_sys
    mat_input = prob.mat_input
    mat_x = prob.x_only_constraint
    mat_u = prob.u_only_constraint

    mat_constraint4w = np.array([[1, 0], [0, 1]])

    constraint_x_u = ConstraintStageXU(
        dim_sys=prob.dim_sys,
        dim_input=prob.dim_input,
        mat_x=mat_x,
        mat_u=mat_u)

    mpctube = MPCqpTube(
        mat_sys, mat_input,
        mat_q=prob.mat_q,
        mat_r=prob.mat_r,
        mat_k_s=prob.mat_k,
        mat_k_z=prob.mat_k,
        mat_constraint4w=mat_constraint4w,
        constraint_x_u=constraint_x_u,
        j_alpha=3,
        alpha_ini=0.01,
        tolerance=0.01)
    horizon = 3
    j_alpha = 3
    mpctube.build_mat_block_ub(horizon=horizon, j_alpha=j_alpha)
    assert mpctube.mat_ub_block.shape[1] == \
        horizon*(prob.dim_input + prob.dim_sys) + prob.dim_sys \
        + j_alpha * prob.dim_sys
    x = np.array([[0.01, 0.01]]).T
    mpctube.build_mat_block_eq(x=x, horizon=horizon)
    mpctube(x, horizon, j_alpha)
