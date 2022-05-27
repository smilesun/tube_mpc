import numpy as np
from tmpc.constraint_eq_ldyn import ConstraintEqLdyn
from tmpc.constraint_block_horizon_terminal import \
    ConstraintBlockHorizonTerminal
from tmpc.constraint_block_horizon_stage_x_u import \
    ConstraintHorizonBlockStageXU
from tmpc.solver_quadprog import quadprog_solve_qp
from tmpc.block_lqr_loss import LqrQpLoss
from tmpc.loss_terminal import LyapunovK





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
                 constraint_x_u,
                 max_iter4pos_inva,
                 tolerance):
        """__init__.
        :param obj_dyn:
        """
        self.constraint_x_u = constraint_x_u
        self.mat_sys = mat_sys
        self.dim_sys = mat_sys.shape[0]
        self.mat_input = mat_input
        self.dim_input = mat_input.shape[1]
        self.mat_q = mat_q
        self.mat_r = mat_r
        #
        self.eq_constraint_builder = ConstraintEqLdyn(
            mat_input=self.mat_input,
            mat_sys=self.mat_sys)
        #
        self.constraint_terminal_block = ConstraintBlockHorizonTerminal(
            constraint_x_u,
            mat_k=mat_k,
            mat_sys=mat_sys,
            mat_input=mat_input,
            max_iter=max_iter4pos_inva,
            tolerance=tolerance)

        self.constraint_stage_block = ConstraintHorizonBlockStageXU(
            mat_state_ub=constraint_x_u.mat_only_x,
            mat_u_ub=constraint_x_u.mat_only_u)

        mat_a_c = self.mat_sys + np.matmul(self.mat_input, mat_k)
        self.mat_p = LyapunovK(mat_a_c, np.eye(self.dim_sys))()

        self.qp_loss = LqrQpLoss(mat_q, mat_r, self.mat_p)  # FIXME: lyapunov function
        # can also be changed dynamically

    def __call__(self, x_obs, horizon):
        """__call__.
        :param x: current state
        """
        dim_sys = x_obs.shape[0]
        mat_dyn_eq, mat_b_dyn_eq = self.eq_constraint_builder(x_obs, horizon)
        block_mat_terminal_a, block_mat_terminal_b = \
            self.constraint_terminal_block(horizon)

        block_mat_stage_a, block_mat_stage_b = \
            self.constraint_stage_block(horizon)

        """
        # there can be more inequality constraint than number of state!
        """
        block_mat_a_ub = np.vstack([block_mat_stage_a,
                                    block_mat_terminal_a])
        """
        [block_m1,
         block_m2,
         block_m3][x_{0:T}, u_{0:T-1}]^T<=[ones(2T+1, 1)^T,
                                           ones(2T+1, 1)^T,
                                           ones(2T+1, 1)^T]^T
        """
        block_mat_b_ub = np.vstack([block_mat_stage_b,
                                    block_mat_terminal_b])

        block_mat_loss = self.qp_loss.gen_loss(self.mat_q, self.mat_r, horizon)
        vec_x_u = quadprog_solve_qp(
            P=block_mat_loss,
            A_ub=block_mat_a_ub,
            b_ub=block_mat_b_ub,
            A_eq=mat_dyn_eq,
            b_eq=mat_b_dyn_eq)
        pos_u = (1+horizon)*dim_sys  # NOTE: no need to minus 1, otherwise wrong!
        vec_u = vec_x_u[pos_u:pos_u+self.dim_input]
        return vec_u.reshape((len(vec_u), 1))
