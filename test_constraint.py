import numpy as np
from constraint_eq_ldyn import ConstraintEqLdyn
from constraint_block_horizon_terminal import ConstraintBlockHorizonTerminal
from constraint_x_u_couple import ConstraintStageXU
from utils_case import Probset


def test_constraint_eq_dyn():
    horizon = 3
    n = 2  # dimension of system
    r = 1
    mat_input = np.zeros((n, r))
    mat_input[0] = 1
    mat_sys = np.eye(n)
    constraint = ConstraintEqLdyn(mat_input, mat_sys)
    x = np.reshape(np.random.rand(2), (n, 1))
    mat, b = constraint(x, horizon)
    mat.shape
    b.shape
    assert mat.shape[1] == b.shape[0]
    assert mat.shape[1] == horizon * (n+r) + n
    assert mat.shape[0] == n*(horizon+1)

def test_constraint_xu():
    mat_x = Probset().x_only_constraint
    mat_u = np.array([[1]])
    obj = ConstraintStageXU(dim_sys=2, dim_input=1,
                            mat_x=mat_x,
                            mat_u=mat_u)

    assert np.all(obj.mat[:mat_x.shape[0], :mat_x.shape[1]] == mat_x)


def test_constraint_terminal():
    prob = Probset()
    mat_x = prob.x_only_constraint
    mat_u = np.array([[1]])
    obj = ConstraintStageXU(prob.dim_sys,
                            prob.dim_input,
                            mat_x=mat_x,
                            mat_u=mat_u)
    horizon = 3
    mat, vec_b = ConstraintBlockHorizonTerminal(obj,
                                                mat_k=np.ones((1,1)),
                                                mat_sys=prob.mat_sys,
                                                mat_input=prob.mat_input)(horizon)
    mat.shape[1] == prob.dim_sys*(horizon + 1) + prob.dim_input*horizon
