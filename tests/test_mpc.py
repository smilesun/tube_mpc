import numpy as np
from tmpc.constraint_x_u_couple import ConstraintStageXU
from tmpc.utils_case import Probset
from tmpc.mpc_qp import MPCqp


def test_mpc():
    horizon = 3

    #
    prob = Probset()
    mat_sys = prob.mat_sys
    mat_input = prob.mat_input
    mat_x = prob.x_only_constraint
    mat_u = prob.u_only_constraint

    constraint_x_u = ConstraintStageXU(
        dim_sys=prob.dim_sys,
        dim_input=prob.dim_input,
        mat_x=mat_x,
        mat_u=mat_u)

    mpc_qp = MPCqp(
        mat_sys, mat_input,
        mat_q=prob.mat_q,
        mat_r=prob.mat_r,
        mat_k=prob.mat_k,
        constraint_x_u=constraint_x_u)

    mpc_qp(prob.x_init, horizon)

