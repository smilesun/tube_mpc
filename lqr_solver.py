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
    def __init__(self, mat_q, mat_r):
        """__init__.
        :param mat_q:
        :param mat_r:
        """
        self.mat_q = mat_q
        self.mat_r = mat_r

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

    def gen_block_terminal_constraint(self, horizon,
                                      mat_ub_inf_pos_inva):
        eye1 = np.zeros((horizon+1, horizon+1))
        eye1[horizon, horizon] = 1
        block_mat_a_ub = np.kron(mat_ub_inf_pos_inva, eye1)
        #
        b_ub1 = np.zeros(horizon)
        b_ub1[horizon, 1] = 1
        b_ub = np.ones(mat_ub_inf_pos_inva.shape[0])
        block_mat_b_ub = np.kron(b_ub, b_ub1)


    def __call__(self, horizon, mat_ub_inf_pos_inva,
                 mat_dyn_eq,  # functional constraint
                 mat_b_dyn_eq  # functional constraint
                 ):
        """__call__."""
        block_mat_loss = self.gen_loss(self.mat_q, self.mat_r, horizon)
        block_terminal_constraint = self.gen_block_terminal_constraint(
            horizon, mat_ub_inf_pos_inva)

        vec_x_u = quadprog_solve_qp(
            P=block_mat_loss,
            A_ub=block_mat_a_ub,
            b_ub=block_mat_b_ub,
            A_eq=mat_dyn_eq,
            b_eq=mat_b_dyn_eq)
        return vec_x_u
