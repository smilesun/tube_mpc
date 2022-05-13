import numpy as np


class ConstraintBlockHorizonTerminal():
    """
    for decision sequence: x_0:T, u_0:T-1
    constraint constraint w.r.t x_T
    """
    def __init__(self, mat_terminal_inf_pos_inva,
                 mat_x_u,
                 mat_u_ub):
        """__init__.
        :param mat_q:
        :param mat_r:
        """
        self.mat_terminal_inf_pos_inva = mat_terminal_inf_pos_inva
        self.mat_u_ub = mat_u_ub
        self.dim_input = mat_u_ub.shape[1]

    def gen_block_terminal_state_inva_constraint(self, horizon):
        """
        positive invariant terminal set only need to apply on x_T
        [0, 0, 0, P, |0, 0, 0] *
        [x_{0:T-1}, x_T,  u_{0:T-1}]^T =Px_T<= ones(nrow(P),1)

        How to generate this P? Note that this P only works on x,
        u is a deterministic funciton of x. So the coupling of
        x and u is based on K^{Riccati}.

        Starting from x_N till infinity, supposed K_{Riccati} is used, then

        - next state must be inside terminal set(positive invariant):
        x^+ = Ax + BKx = (A+BK)x must be inside positive invariant set
        this is satisfied by the positive invariant set with A_c = A+BK as
        closed loop system matrix. That means, P is the positive invariant
        set w.r.t. A_c=A+BK

        - K^{Riccati}*x must be inside the control input constraint d^T u <=1:
            i.e.   for x_{T:inft}: D(Kx) <=1, how to ensure?

            - Note Cx+Du<=1 is the general coupled state-control constraint,
            [C, D] can have multiple block rows.
            - One row can be
                [0, d^T][x, u]^T =d^T u<=1
            - Another row can be
                [c^T, 0][x^T,u^T]^T = c^T x <=1

        - Since u=kx (only after x_T, inside the horizon, the control signal
        can be arbitrary), this is equivalent to d^TKx<=1, this is new
        constraint to $x$, in addition to c^T x <=1
        - in general,  Cx+Du<=1 can be transformed to
          Cx+DKx = [CI, DK]x <=1 : since u is uniquely defined by $x$, all
          constraints on u can be represented by $x$.
          so the constraint on $X$ should be:
          - Cx+DKx = [CI, DK]x <=1
          - Will the constraint [c^T, 0][x^T, u^T]^T = c^Tx <=1
            be covered by [CI, DK]x=[C,D][I, K^T]^T<=1 ?
            i.e. is the following two rows redundant w.r.t. each other?
            [c^T, 0][I, K^T]^T x <= 1
            [c^T, 0] x <= 1
            Yes! since [c^T, 0][I, K^T]^T x = c^T x <=1

        - Conclusion: the constraint for state will be altered by K^{Riccati}
          Cx+DKx = [CI, DK]x <=1 : since u is uniquely defined by $x$
        """
        block_one_hot = np.zeros((1, 2*horizon+1))
        block_one_hot[horizon] = 1
        block_mat_terminal_ub = np.kron(
            self.mat_terminal_inf_pos_inva, block_one_hot)
        block_b_terminal_ub = np.ones(
            self.mat_terminal_inf_pos_inva.shape[0], 1)
        return block_mat_terminal_ub, block_b_terminal_ub

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
