import numpy as np
from constraint_eq_ldyn import ConstraintEqLdyn
from constraint_block_horizon_terminal import ConstraintBlockHorizonTerminal
from constraint_x_u_couple import ConstraintStageXU


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
    mat_x = np.array(
        [[2, -1],
         [0, 1],
         [1, 0],
         [0, -1]])

    mat_u = np.array([[1]])

    obj = ConstraintStageXU(dim_sys=2, dim_input=1,
                            mat_x=mat_x,
                            mat_u=mat_u)

    assert np.all(obj.mat[:mat_x.shape[0], :mat_x.shape[1]] == mat_x)
