import numpy as np
import scipy
from tmpc.solver_quadprog import quadprog_solve_qp
from tmpc.constraint_block_horizon_terminal import ConstraintBlockHorizonTerminal
from tmpc.constraint_block_horizon_stage_x_u import ConstraintHorizonBlockStageXU


def qp_x_u_1step(mat_q, mat_r, mat_dyn_eq, mat_b_dyn_eq, mat_ub_inf_pos_inva):
    """
    decision variable: d=[x_{1:T}, u_{1:T}]^T
    Loss: x^TQx+u^TRu = d^T[[Q,0], [0, R]]d
    """
    mat_loss = np.block([[mat_q, np.zeros((mat_q.shape[0], mat_r.shape[1]))],
                         [np.zeros((mat_r.shape[0], mat_q.shape[1])), mat_r]])

    vec_x_u = quadprog_solve_qp(P=mat_loss,
                                A_ub=mat_ub_inf_pos_inva,
                                b_ub=np.ones(mat_ub_inf_pos_inva.shape[0]),
                                A_eq=mat_dyn_eq,
                                b_eq=mat_b_dyn_eq)
    return vec_x_u


class LqrQp():
    """LqrQp.
    solving LQR problem with quadratic programming
    """
    def __init__(self, mat_q, mat_r,
                 mat_k,
                 mat_sys, mat_input,
                 constraint_x_u):
        """__init__.
        :param mat_q:
        :param mat_r:
        """
        self.mat_q = mat_q
        self.mat_r = mat_r
        self.dim_sys = self.mat_q.shape[0]
        self.dim_input = self.mat_r.shape[0]
        self.constraint_terminal = ConstraintBlockHorizonTerminal(
            constraint_x_u,
            mat_k=mat_k,
            mat_sys=mat_sys,
            mat_input=mat_input)
        self.constraint_stage = ConstraintHorizonBlockStageXU(
            mat_state_ub=constraint_x_u.mat_x,
            mat_u_ub=constraint_x_u.mat_u)

    def gen_loss_block_q(self, mat_q, horizon):
        """
        - loss should be dynamically changed
        - suppose horizon is T=3, decision variable x_{0:T}
        0, 0, 0, 0
        0, Q, 0, 0
        0, 0, Q, 0
        0, 0, 0, Q
        """
        eye1 = np.eye(horizon+1)
        eye1[0, 0] = 1.0
        return np.kron(mat_q, eye1)

    def gen_loss_block_r(self, mat_r, horizon):
        """
        suppose horizon is T=3, decision variable x_{0:T}
        R, 0, 0
        0, R, 0
        0, 0, R
        """
        return np.kron(mat_r, np.eye(horizon))

    def gen_loss(self, mat_q, mat_r, horizon):
        block_q = self.gen_loss_block_q(mat_q, horizon)
        block_r = self.gen_loss_block_r(mat_r, horizon)
        return scipy.linalg.block_diag(block_q, block_r)

    def __call__(self, horizon,
                 mat_dyn_eq,  # functional constraint
                 mat_b_dyn_eq  # functional constraint
                 ):
        """__call__."""
        block_mat_loss = self.gen_loss(self.mat_q, self.mat_r, horizon)

        block_mat_terminal_a, block_mat_terminal_b = \
            self.constraint_terminal(horizon)

        block_mat_stage_a, block_mat_stage_b = \
            self.constraint_stage(horizon)

        """
        # there can be more inequality constraint than number of state!
        """
        block_mat_a_ub = block_mat_terminal_a
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
        vec_x_u = quadprog_solve_qp(
            P=block_mat_loss,
            A_ub=block_mat_a_ub,
            b_ub=block_mat_b_ub,
            A_eq=mat_dyn_eq,
            b_eq=mat_b_dyn_eq)
        return vec_x_u
