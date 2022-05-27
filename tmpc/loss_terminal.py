from scipy import linalg


class LyapunovK():
    """
    V(x) = 1/2 x^T P x
    V(x_{t}) - V(x_{t+1})
    =  1/2 x_t^T P x_t - 1/2 x_{t+1}^T P x_{t+1}
    =  1/2 x_t^T P x_t - 1/2 x_{t}^T A^T P A x_t
    =  1/2 x^t^T (P - A^TPA)x_t
    For V(x_{t}) - V(x_{t+1}) > 0
    A^TPA-P=-Q, s.t. Q > 0
    """
    def __init__(self, mat_a_c, mat_q):
        """__init__.
        :param mat_a_c: A in A^TPA-P<-Q
        :param mat_q: Q in A^TPA-P<-Q
        """
        self.mat_a_c = mat_a_c
        self.mat_q = mat_q

    def __call__(self):
        mat_p = linalg.solve_discrete_lyapunov(self.mat_a_c.T, self.mat_q)
        return mat_p
