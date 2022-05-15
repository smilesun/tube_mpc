import numpy as np
from constraint_pos_inva_terminal import PosInvaTerminalSetBuilder
from constraint_eq_ldyn import ConstraintEqLdyn
from constraint_block_horizon_lqr_solver import LqrQp


class MPCqp():
    """
    T: horizon of planning where T is the terminal point
    dim_s: dimension of dynamic system
    Dynamic (Equality) Constraint:
        [diag[A]_{T-1}, [0]_{T-1*dim_s}, diag[B]_{T-1}}]

        :dim(diag[A]_{T-1}) = (dim_s * (T-1)) * (dim_s * (T-1))
        :dim(diag[B]_{T-1}}) = (dim_s*(T-1)) * (dim_u * (T-1))
    suppose T = 3
    decision variables:  u_0, u_1, u_2=u_{T-1}
                         x_1, x_2, x_3=x_{T}
    input: current state x_0=x
    equality constraint for input and decision variable: Ax_0 + Bu_0 = x_1
    ........................
    Inequality constraint
    [0, 0, M_{inf}, 0, 0] x < 1
    """
    def __init__(self, mat_sys, mat_input,
                 mat_q, mat_r, mat_k,
                 constraint_x_u):
        """__init__.
        :param obj_dyn:
        """
        self.constraint_x_u = constraint_x_u
        self.mat_sys = mat_sys
        self.mat_input = mat_input
        self.dim_input = mat_input.shape[1]
        self.mat_q = mat_q
        self.mat_r = mat_r
        #
        self.eq_constraint_builder = ConstraintEqLdyn(
            mat_input=self.mat_input,
            mat_sys=self.mat_sys)
        #
        self.lqr_solver = LqrQp(
            constraint_x_u=self.constraint_x_u,
            mat_q=self.mat_q,
            mat_r=self.mat_r,
            mat_k=mat_k,
            mat_sys=self.mat_sys,
            mat_input=self.mat_input)

    def __call__(self, x_obs, horizon):
        """__call__.
        :param x: current state
        """
        dim_sys = x_obs.shape[0]
        mat_dyn_eq, mat_b_dyn_eq = self.eq_constraint_builder(x_obs, horizon)
        vec_x_u = self.lqr_solver(
            horizon,
            mat_dyn_eq,  # functional constraint
            mat_b_dyn_eq  # functional constraint
            )
        pos_u = (1+horizon)*dim_sys-1
        return vec_x_u[pos_u:pos_u+self.dim_input, 1]
