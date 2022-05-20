import numpy as np
import scipy
from tmpc.solver_quadprog import quadprog_solve_qp


class LqrQpLoss():
    """
    """
    def __init__(self, mat_q, mat_r):
        """__init__.
        :param mat_q:
        :param mat_r:
        """
        self.mat_q = mat_q
        self.mat_r = mat_r
        self.dim_sys = self.mat_q.shape[0]
        self.dim_input = self.mat_r.shape[0]

    def gen_loss_block_q(self, mat_q, horizon):
        """
        - loss should be dynamically changed, so add arguments
        - suppose horizon is T=3, decision variable x_{0:T}
        0, 0, 0, 0
        0, Q, 0, 0
        0, 0, Q, 0
        0, 0, 0, Q
        """
        eye1 = np.eye(horizon+1)
        eye1[0, 0] = 0.0
        # NOTE: not return np.kron(mat_q, eye1)
        return np.kron(eye1, mat_q)

    def gen_loss_block_r(self, mat_r, horizon):
        """
        suppose horizon is T=3, decision variable x_{0:T}
        R, 0, 0
        0, R, 0
        0, 0, R
        """
        # NOTE: not return np.kron(mat_r, np.eye(horizon))
        return np.kron(np.eye(horizon), mat_r)

    def gen_loss(self, mat_q, mat_r, horizon, j_alpha):
        block_q = self.gen_loss_block_q(mat_q, horizon)
        block_r = self.gen_loss_block_r(mat_r, horizon)
        block_w = self.gen_loss_block_w(mat_q, j_alpha)
        return scipy.linalg.block_diag(block_q, block_r)

    def gen_loss_block_w(self, mat_q, j_alpha):
        """
        - loss should be dynamically changed, so add arguments
        - suppose horizon is T=3, decision variable x_{0:T}
        0, 0, 0, 0
        0, Q, 0, 0
        0, 0, Q, 0
        0, 0, 0, Q
        """
        eye1 = np.eye(j_alpha)
        eye1[0, 0] = 0.0
        # NOTE: not return np.kron(mat_q, eye1)
        return np.kron(eye1, np.zeros((mat_q.shape[0], mat_q.shape[0])))
