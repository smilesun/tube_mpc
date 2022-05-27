import numpy as np
import scipy
from tmpc.solver_quadprog import quadprog_solve_qp


class LqrQpLoss():
    """
    """
    def __init__(self, mat_q, mat_r, mat_p):
        """__init__.
        :param mat_q:
        :param mat_r:
        :param mat_p: P is the 1/2 x^TPx Lyapunov function term
        """
        self.mat_q = mat_q
        self.mat_r = mat_r
        self.dim_sys = self.mat_q.shape[0]
        self.dim_input = self.mat_r.shape[0]
        self._mat_p = mat_p

    @property
    def mat_p(self):
        """
        P is the 1/2 x^TPx Lyapunov function term
        """
        return self._mat_p

    def gen_loss_block_q(self, mat_q, horizon):
        """
        - loss should be dynamically changed, so add arguments
        - suppose horizon is T=3, decision variable x_{0:T}
        0, 0, 0, 0
        0, Q, 0, 0
        0, 0, Q, 0
        0, 0, 0, P
        where P is the 1/2 x^TPx Lyapunov function term
        """
        eye1 = np.eye(horizon+1)
        eye1[0, 0] = 0.0
        # NOTE: not return np.kron(mat_q, eye1)
        block_q = np.kron(eye1, mat_q)
        block_q[horizon*self.dim_sys:, horizon*self.dim_sys:] = self.mat_p
        return block_q

    def gen_loss_block_r(self, mat_r, horizon):
        """
        suppose horizon is T=3, decision variable x_{0:T}
        R, 0, 0
        0, R, 0
        0, 0, R
        """
        # NOTE: not return np.kron(mat_r, np.eye(horizon))
        return np.kron(np.eye(horizon), mat_r)

    def gen_loss(self, mat_q, mat_r, horizon):
        block_q = self.gen_loss_block_q(mat_q, horizon)
        block_r = self.gen_loss_block_r(mat_r, horizon)
        return scipy.linalg.block_diag(block_q, block_r)


class LqrQpLossTube(LqrQpLoss):
    def gen_loss(self, mat_q, mat_r, horizon, j_alpha):
        block_q = self.gen_loss_block_q(mat_q, horizon)
        block_r = self.gen_loss_block_r(mat_r, horizon)
        block_w = self.gen_loss_block_w(mat_q, j_alpha)
        return scipy.linalg.block_diag(block_q, block_r, block_w)

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
