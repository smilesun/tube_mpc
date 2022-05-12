import numpy as np
from constraint_eq_ldyn import ConstraintEqLdyn

def test_constraint_nominal_kron():
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
