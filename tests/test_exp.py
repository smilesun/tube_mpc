import numpy as np
import control
from tmpc.mpc_qp_tube import MPCqpTube
from tmpc.constraint_x_u_couple import ConstraintStageXU
from tmpc.utils_case import Probset
from tmpc.simulate import Exp
from tmpc.dyn_sys import DynSysL


def test_exp():
    horizon = 8
    prob = Probset()

    constraint_x_u = ConstraintStageXU(
        dim_sys=prob.dim_sys,
        dim_input=prob.dim_input,
        mat_x=prob.x_only_constraint,
        mat_u=prob.u_only_constraint)

    mat_k_s = control.place(prob.mat_sys, prob.mat_input, [-0.2, 0.1])
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
        alpha_ini=1e-6,
        tolerance=1e-5)

    x = np.array([[-0.8, 2]]).T
    dyn = DynSysL(dim_sys=prob.dim_sys,
                  dim_u=prob.dim_input,
                  x_ini=x,
                  constraint_x_u=constraint_x_u,
                  max_w=prob.max_w,
                  mat_sys=prob.mat_sys,
                  mat_input=prob.mat_input)
    exp = Exp(dyn, controller=mpctube)
    print("new exp")
    exp.run(20, horizon)
    exp.run(200, horizon)
