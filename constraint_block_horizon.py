import numpy as np
from solver_quadprog import quadprog_solve_qp


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
        return np.diag([block_q, block_r])

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
        - k_riccati*x must be inside the control input
        control constraint: for x_{T:inft}: C(Kx) <=1, how to ensure?
        - next state must be inside terminal set(positive invariant):
        x^+ = Ax + BKx = (A+BK)x must be inside positive invariant set
        """
        pass

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
        block_mat_a_ub = np.vstack([block_mat_ub_input_a, block_mat_ub_input_b])
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
