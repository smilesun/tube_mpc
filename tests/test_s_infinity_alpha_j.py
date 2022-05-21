import numpy as np
from tmpc.mpc_qp_tube import MPCqpTube
from tmpc.constraint_x_u_couple import ConstraintStageXU
from tmpc.utils_case import Probset
from tmpc.constraint_s_inf import ConstraintSAlpha


def test_j_alpha():
    prob = Probset()
    mat_constraint4w = np.array([[1, 0], [0, 1]])
    constraint_j_alpha = ConstraintSAlpha(
        mat_sys=prob.mat_sys,
        mat_input=prob.mat_input,
        mat_k_s=prob.mat_k,
        mat_w=mat_constraint4w)
    constraint_j_alpha.cal_power_given_alpha(0.01)
    constraint_j_alpha.cal_alpha_given_power(3)
