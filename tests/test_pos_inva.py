import numpy as np
from tmpc.constraint_block_horizon_terminal import ConstraintBlockHorizonTerminal
from tmpc.utils_case import Probset
from tmpc.constraint_x_u_couple import ConstraintStageXU


def test_constraint_terminal():
    prob = Probset()
    obj = ConstraintStageXU(prob.dim_sys,
                            prob.dim_input,
                            mat_x=prob.x_only_constraint,
                            mat_u=prob.u_only_constraint)
    horizon = 3
    # [A_{n*n}+B_{n*r}K_{r*n}]X_{n*1}
    constraint_terminal = ConstraintBlockHorizonTerminal(
        obj,
        # NOTE: not arbitrary k! mat_k=np.ones((1, 2)),
        mat_k=prob.mat_k,
        mat_sys=prob.mat_sys,
        mat_input=prob.mat_input,
        max_iter=10,
        tolerance=1e-7)
    mat, vec_b = constraint_terminal(horizon)
    assert mat.shape[1] == prob.dim_sys*(horizon + 1) + prob.dim_input*horizon
