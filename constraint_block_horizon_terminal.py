import numpy as np


class ConstraintBlockHorizonTerminal():
    """
    for decision sequence: x_0:T, u_0:T-1
    constraint constraint w.r.t x_T
    """
    def __init__(self, mat_terminal_inf_pos_inva,
                 mat_x_u,
                 mat_state_ub,
                 mat_u_ub):
        """__init__.
        :param mat_q:
        :param mat_r:
        """
        self.mat_terminal_inf_pos_inva = mat_terminal_inf_pos_inva
        self.mat_u_ub = mat_u_ub
        self.mat_state_ub = mat_state_ub
        self.dim_sys = mat_state_ub.shape[1]
        self.dim_input = mat_u_ub.shape[1]

    def gen_block_terminal_state_inva_constraint(self, horizon):
        """
        positive invariant terminal set only need to apply on x_T
        [0, 0, 0, P, |0, 0, 0] *
        [x_{0:T-1}, x_T,  u_{0:T-1}]^T =Px_T<= ones(nrow(P),1)
        """
        block_one_hot = np.zeros((1, 2*horizon+1))
        block_one_hot[horizon] = 1
        block_mat_terminal_ub = np.kron(
            self.mat_terminal_inf_pos_inva, block_one_hot)
        block_b_terminal_ub = np.ones(
            self.mat_terminal_inf_pos_inva.shape[0], 1)
        return block_mat_terminal_ub, block_b_terminal_ub

    def gen_block_terminal_state_control_coupling(self, horizon, k_riccati):
        """
        starting from x_N till infinity, supposed k_riccati is used, then
        - next state must be inside terminal set(positive invariant):
        x^+ = Ax + BKx = (A+BK)x must be inside positive invariant set
        this is satisfied by the positive invariant set
        - k_riccati*x must be inside the control input
        control constraint: for x_{T:inft}: C(Kx) <=1, how to ensure?
        since x_{N+1} = Ax_T + BKx_T = (A+BK)x_T, so
        C(A+BK)x_T<=1
        """
        pass

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
