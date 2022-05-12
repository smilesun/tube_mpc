import numpy as np
from constraint_nominal_kron import ConstraintEQLdyn

def test_constraint_nominal_kron():
    n = 2  # dimension of system
    r = 1
    mat_input = np.zeros((n, r))
    mat_input[0] = 1
    mat_sys = np.eye(n)
    ConstraintEq()
