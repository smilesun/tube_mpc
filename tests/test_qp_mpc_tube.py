import numpy as np
from tmpc.mpc_qp_tube import MPCqpTube
from tmpc.constraint_x_u_couple import ConstraintStageXU
from tmpc.utils_case import ScenarioDummy
import control


def test_tube():
    prob = ScenarioDummy()

    constraint_x_u = ConstraintStageXU(
        dim_sys=prob.dim_sys,
        dim_input=prob.dim_input,
        mat_x=prob.x_only_constraint,
        mat_u=prob.u_only_constraint)

    mat_k_s = control.place(prob.mat_sys, prob.mat_input, [-0.2, -0.1])
    mat_k_s = -1.0 * mat_k_s

    mpctube = MPCqpTube(
        mat_sys=prob.mat_sys,
        mat_input=prob.mat_input,
        mat_q=prob.mat_q,
        mat_r=prob.mat_r,
        mat_k_s=mat_k_s,
        mat_k_z=mat_k_s,
        mat_constraint4w=prob.mat_w,
        constraint_x_u=constraint_x_u,
        alpha_ini=0.01,
        tolerance=0.01)
    horizon = 3
    mpctube.build_mat_block_ub(horizon=horizon)
    assert mpctube.mat_ub_block.shape[1] == \
        horizon*(prob.dim_input + prob.dim_sys) + prob.dim_sys \
        + mpctube.j_alpha * prob.dim_sys
    x = np.array([[0.01, 0.01]]).T
    mpctube.build_mat_block_eq(x=x, horizon=horizon)
    mpctube(x, horizon)
