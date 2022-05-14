import numpy as np
from constraint_x_u_couple import ConstraintStageXU
from constraint_pos_inva_terminal import PosInvaTerminalSetBuilder


class ConstraintBlockHorizonTerminal():
    """
    - for decision sequence: x_0:T, u_0:T-1, generate constraint w.r.t x_T in
    block matrix form  w.r.t all 2*T + 1  decision variables.
    - input: stage wise coupled x (state) and u (control signal) constraint
    C^T x + D^Tu <=1 can be several rows
    first row:
        c1x + 0u <=1 will be [c1_{nrow(c1)*n}, 0_{nrow(c1)*r] [x^T, u^T]^T <=1
    second row:
        0x + d2u <=1 will be [0_{nrow(d1)*n}, d2_{nrow(d1)*r] [x^T, u^T]^T <=1
    third row:
        c3x + d3u <=1 will be [0_{nrow(d1)*n}
    Summarizing the above constraint:
        M_y y <=1
        where
        M_y = [[c1, 0], [0, d2], [c3, d3]]
        and y = [x, u]
    """
    def __init__(self, constraint_x_u, mat_k, mat_sys, mat_input):
        """__init__.
        :param mat_q:
        """
        mat_state_constraint = constraint_x_u.reduce2x(mat_k)
        mat_sys_closed_loop = mat_sys + np.matmul(mat_input, mat_k)  # A+BK
        self.pos_inva = PosInvaTerminalSetBuilder(
            mat_sys_closed_loop,
            mat_state_constraint)
        self.mat_term_inf_pos_inva_k = self.pos_inva(3)  # FIXME:

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

        positive invariant terminal set only need to apply on x_T
        [0, 0, 0, P, |0, 0, 0] *
        [x_{0:T-1}, x_T,  u_{0:T-1}]^T =Px_T<= ones(nrow(P),1)
        P = pos_inva([CI, DK])
        To decouple, P should be precalculated by other routine and feed into
        this class directly.
        """

        block_one_hot = np.zeros((1, 2*horizon+1))
        block_one_hot[0, horizon] = 1
        block_mat_terminal_ub = np.kron(
            self.mat_term_inf_pos_inva_k, block_one_hot)
        block_b_terminal_ub = np.ones(
            (self.mat_term_inf_pos_inva_k.shape[0], 1))
        return block_mat_terminal_ub, block_b_terminal_ub

    def __call__(self, horizon):
        """__call__."""
        block_mat_terminal_a, block_mat_terminal_b = \
            self.gen_block_terminal_state_inva_constraint(horizon)
        return block_mat_terminal_a, block_mat_terminal_b
