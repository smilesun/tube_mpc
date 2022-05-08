import numpy as np
from solver_quadprog import quadprog_solve_qp

def qp_x_u(mat_q, mat_r, mat_dyn_eq, mat_b_dyn_eq, mat_inf_pos_inva):
    """
    decision variable: [x_{1:N}, u_{1:N}]^T
    """
    P = np.block([[mat_q, np.zeros((mat_q.shape[0], mat_r.shape[1]))],
                  [np.zeros((mat_r.shape[0], mat_q.shape[1])), mat_r]])

    quadprog_solve_qp(P=P, A_ub=mat_inf_pos_inva,
                      b_ub=np.ones(mat_inf_pos_inva.shape[0]))


class QP_MPC():
    """
    N: horizon of planning where N is the terminal point
    dim_s: dimension of dynamic system
    Dynamic (Equality) Constraint:
        [diag[A]_{N-1}, [0]_{N-1*dim_s}, diag[B]_{N-1}}]

        :dim(diag[A]_{N-1}) = (dim_s * (N-1)) * (dim_s * (N-1))
        :dim(diag[B]_{N-1}}) = (dim_s*(N-1)) * (dim_u * (N-1))
    suppose N = 2
    [[A]_s, [0]_s, [0]_s
    """
    def __init__(self, obj_dyn):
        """__init__.
        :param obj_dyn:
        """
        self.obj_dyn = obj_dyn

    def __call__(self, x):
        """__call__.
        :param x: current state
        """
        pass
