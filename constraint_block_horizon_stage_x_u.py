import numpy as np
from solver_quadprog import quadprog_solve_qp


class ConstraintHorizonBlockStageXU():
    """LqrQp.
    solving LQR problem with quadratic programming
    """
    def __init__(self, mat_terminal_inf_pos_inva,
                 mat_state_ub,
                 mat_u_ub,
                 mat_q, mat_r):
        """__init__.
        :param mat_q:
        :param mat_r:
        """
        self.mat_terminal_inf_pos_inva = mat_terminal_inf_pos_inva
        self.mat_q = mat_q
        self.mat_r = mat_r
        self.mat_u_ub = mat_u_ub
        self.mat_state_ub = mat_state_ub
        self.dim_sys = self.mat_q.shape[0]
        self.dim_input = self.mat_r.shape[0]

    def gen_block_control_stage_constraint(self, horizon):
        """
        # stage control constraint
        Cu_0<=1
        Cu_1<=1
        Cu_T<=1:
        C_block=
        [0, 0, 0, 0, |C, 0, 0] [x_{0:T}, u_{0:T-1}]^T =Cu_0<= ones(nrow(C),1)
        [0, 0, 0, 0, |0, C, 0] [x_{0:T}, u_{0:T-1}]^T =Cu_1<= ones(nrow(C),1)
        [0, 0, 0, 0, |0, 0, C] [x_{0:T}, u_{0:T-1}]^T =Cu_{T-1}<= ones(nrow(C),1)
        C_block[x_{0:T}, u_{0:T-1}]^T <= ones(T*nrow(C), 1)
        """
        nrow = self.mat_u_ub.shape[0]
        zeros = np.zeros((horizon*nrow, (horizon+1)*self.dim_sys))
        block_mat_diag = np.kron(self.mat_u_ub, np.eye(horizon))
        block_mat4u = np.hstack([zeros, block_mat_diag])
        #
        b_ub_global4u = np.ones((horizon*self.mat_u_ub.shape[0], 1))
        return block_mat4u, b_ub_global4u

    def build_block_state_constraint(self, horizon):
        """
        Mx<=1
        Mx_0<=1
        Mx_T<=1:
        M_block=
        [0, 0, 0, 0, |0, 0, 0 # this line does not has to be added
        ---------------------------------------------------------
        [0, M, 0, 0, |0, 0, 0] [x_{0:T}, u_{0:T-1}]^T =Mx_1<= ones(nrow(M),1)
        [0, 0, M, 0, |0, 0, 0] [x_{0:T}, u_{0:T-1}]^T =Mx_2<= ones(nrow(M),1)
        [0, 0, 0, M, |0, 0, 0] [x_{0:T}, u_{0:T-1}]^T =Mx_T<= ones(nrow(M),1)
        M_block[x_{0:T}, u_{0:T-1}]^T <= ones(dim(x)*T, 1)
        """
        mat_block = np.eye(horizon+1)
        mat_block[0, 0] = 0
        block_mat_ub = np.kron(self.mat_state_ub, mat_block)
        block_mat_ub = block_mat_ub[1:, :]  # drop the first row (all zero)
        """
        [0]_{nrow(M)[0], r}, [0]_{nrow(M)[0], r}, [0]_{nrow(M)[0], r}
        [0]_{nrow(M)[0], r}, [0]_{nrow(M)[0], r}, [0]_{nrow(M)[0], r}
        [0]_{nrow(M)[0], r}, [0]_{nrow(M)[0], r}, [0]_{nrow(M)[0], r}
        """
        block_zero = np.zeros((self.mat_state_ub.shape[0]*horizon,
                               self.dim_input*horizon))
        # block_zero = np.kron(np.zeros((horizon, horizon)),
        block_mat_ub = np.hstack([block_mat_ub, block_zero])
        block_b_ub = np.ones((self.mat_state_ub.shape[0]*horizon, 1))
        return block_mat_ub, block_b_ub

    def __call__(self, horizon,
                 mat_dyn_eq,  # functional constraint
                 mat_b_dyn_eq  # functional constraint
                 ):
        """__call__."""
        block_mat_loss = self.gen_loss(self.mat_q, self.mat_r, horizon)
        block_mat_terminal_a, block_mat_terminal_b = \
            self.gen_block_terminal_state_inva_constraint(horizon)
        block_mat_ub_state_a, block_mat_ub_state_b = \
            self.build_block_state_constraint(horizon)
        block_mat_ub_input_a, block_mat_ub_input_b = \
            self.gen_block_control_stage_constraint(horizon)
        """
        # there can be more inequality constraint than number of state!
        """
        block_mat_a_ub = block_mat_terminal_a
        block_mat_a_ub = np.vstack([block_mat_ub_state_a, block_mat_a_ub])
        block_mat_a_ub = np.vstack(
            [block_mat_ub_input_a, block_mat_ub_input_b])
        """
        [block_m1,
         block_m2,
         block_m3][x_{0:T}, u_{0:T-1}]^T<=[ones(2T+1, 1)^T,
                                           ones(2T+1, 1)^T,
                                           ones(2T+1, 1)^T]^T
        """
        #
        block_mat_b_ub = block_mat_terminal_b
        block_mat_b_ub = np.vstack([block_mat_ub_state_b, block_mat_b_ub])
        # block_mat_b_ub = np.vstack([block_mat_input_b, block_mat_b_ub])

        vec_x_u = quadprog_solve_qp(
            P=block_mat_loss,
            A_ub=block_mat_a_ub,
            b_ub=block_mat_b_ub,
            A_eq=mat_dyn_eq,
            b_eq=mat_b_dyn_eq)
        return vec_x_u
