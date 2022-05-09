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
    suppose N = 3
    decision variables:  u_0, u_1, u_2=u_{N-1}
                         x_1, x_2, x_3=x_{N}
    input: current state x_0=x
    equality constraint for input and decision variable: Ax_0 + Bu_0 = x_1
    where input is x=x_0, decision variable is u_0, x_1:
    let x=[x_1, x_2, x_3=x_N, u_0, u_1, u_2=u_{N-1}
    then,
    [-I_{d},[0]_d,[0]_d,  B_{d*r],[0]_{d*r],[0]_{d*r]] x = A_{d*d}[x_0]_{d*1}
    The rest equality constraint:
        Ax_k + Bu_k = x_{k+1}:
    Ax_1 + Bu_1 = x_2
    Ax_2 + Bu_2 = x_3
    [[A]_d,     -I_d,  [0]_{d*d} | [B]_{d*r}, [0]_{d*r}, [0]_{d*r} ]x = [0]_{d*1}
    [[0]_{d*d}, [A]_d, -I_d      | [0]_{d*r}, [B]_{d*r}, [0]_{d*r} ]x = [0]_{d*1}
    ........................
    Inequality constraint
    [0, 0, M_{inf}, 0, 0] x < 1
    """
    def __init__(self, obj_dyn, eq_constraint_builder):
        """__init__.
        :param obj_dyn:
        """
        self.obj_dyn = obj_dyn
        self.eq_constraint_builder = eq_constraint_builder

    def __call__(self, x):
        """__call__.
        :param x: current state
        """
        A_eq, b_eq = self.eq_constraint_builder()
        # A_ub, b_ub = self.ub_constraint_builder()
