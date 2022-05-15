import numpy as np
from constraint_eq_ldyn import ConstraintEqLdyn
from constraint_block_horizon_terminal import ConstraintBlockHorizonTerminal
from constraint_block_horizon_stage_x_u import ConstraintHorizonBlockStageXU
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
    assert mat.shape[0] == b.shape[0]
    # x0=x
    # x1=Ax0+bu0
    # x2=
    # x3
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
    # [A_{n*n}+B_{n*r}K_{r*n}]X_{n*1}
    constraint_terminal = ConstraintBlockHorizonTerminal(
        obj,
        mat_k=np.ones((1, 2)),
        mat_sys=prob.mat_sys,
        mat_input=prob.mat_input)
    mat, vec_b = constraint_terminal(horizon)
    assert mat.shape[1] == prob.dim_sys*(horizon + 1) + prob.dim_input*horizon

def test_constraint_xu_stage_block():
    prob = Probset()
    mat_x = prob.x_only_constraint
    mat_u = np.array([[1]])
    obj = ConstraintStageXU(dim_sys=prob.dim_sys,
                            dim_input=prob.dim_input,
                            mat_x=mat_x,
                            mat_u=mat_u)
    constra = ConstraintHorizonBlockStageXU(mat_state_ub=obj.mat_x,
                                  mat_u_ub=obj.mat_u)
    mat, vec_b = constra(3)
