import numpy as np
import control
from tmpc.constraint_x_u_couple import ConstraintStageXU
from tmpc.utils_case import Probset
from tmpc.mpc_qp import MPCqp
from tmpc.simulate import Exp
from tmpc.dyn_sys import DynSysL


def test_mpc():
    horizon = 3

    #
    prob = Probset()

    constraint_x_u = ConstraintStageXU(
        dim_sys=prob.dim_sys,
        dim_input=prob.dim_input,
        mat_x=prob.x_only_constraint,
        mat_u=prob.u_only_constraint)

    mat_k_s = control.place(prob.mat_sys, prob.mat_input, [-0.2, 0.1])
    mat_k_s = -1.0 * mat_k_s

    mpc_qp = MPCqp(
        prob.mat_sys,
        prob.mat_input,
        mat_q=prob.mat_q,
        mat_r=prob.mat_r,
        mat_k=mat_k_s,
        constraint_x_u=constraint_x_u)

    mpc_qp(prob.x_init, horizon)

    dyn = DynSysL(dim_sys=prob.dim_sys,
                  dim_u=prob.dim_input,
                  x_ini=prob.x_init,
                  constraint_x_u=constraint_x_u,
                  max_w=0,
                  mat_sys=prob.mat_sys,
                  mat_input=prob.mat_input)
    exp = Exp(dyn, controller=mpc_qp)
    print("new exp")
    exp.run(20, horizon)
