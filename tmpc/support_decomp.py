"""
## find finite sequence approximation of S_{\\infty} using spectral radius,
conditioned on approximation parameter $\\alpha$
$S_{\\infty} \\approx  \\hat{S, \\alpha}$
\\hat{S, \\alpha}$
=$\\minkowski_sum_{i=0:k_{\\alpha}-1}(A+BK^{s})^i*W$ where \\alpha has
relation to the spectral radius

## find Decomposition of support function w.r.t. \\hat{S, \\alpha}:
- h(\\hat{S}, q)= \\sum_{j=0:k_{\\alpha}-1}h(W, [(A+BK^{s})^j]^T*q)
"""
import numpy as np
from tmpc.support_fun import fun_support


class SupportDecomp():
    """
    - set S={minkowski_sum}_{i=0:k_{\\alpha}-1}S_i
      h(S, q) = \\sum_{i=0:k_{\\alpha}-1}h(S_i, q)
    - set S=A*W={x=Aw, s.t. w \\in W}
      h(S,q)
      = h(A*W, q) = max_{x\\in A*W}{x^Tq, s.t. x=Aw, s.t. w \\in W}
      = max_{w\\in W} (Aw)^Tq = max_{w\\in W} (w^TA^Tq)
      = h(W, A^Tq)
    ## set \\hat{S, \\alpha}={minkowski_sum}_{i=0:k_{\\alpha}-1}(A+BK^{s})^i*W$
    where \\alpha has relation to the spectral radius

    ## find Decomposition of support function w.r.t. \\hat{S, \\alpha}:
    - h(\\hat{S}, q)= \\sum_{j=0:k_{\\alpha}-1}h(W, [(A+BK^{s})^j]^T*q)
    """
    def __init__(self, mat_set, mat_sys, mat_input, mat_k_s):
        """__init__.
        :param mat_set:
        :param mat_sys:
        :param mat_input:
        :param mat_k_s:
        """
        self.mat_set = mat_set
        self.mat_a_c = mat_sys + np.matmul(mat_input, mat_k_s)
        self.mat_a_c_power = np.eye(self.mat_a_c.shape[0])

    def fun_support_decomp(self, vec_q, j_power):
        """fun_support_decomp.

        :param vec_q:
        :param j_power:
        """
        mat_a_c_j = np.linalg.matrix_power(self.mat_a_c, j_power)
        vec_direction = np.matlmul(mat_a_c_j.T, vec_q)
        return fun_support(self.mat_set, vec_direction, b_ub=None)

    def fun_support_minkow_sum(self, vec_q, k_alpha):
        """fun_support_minkow_sum.

        :param vec_q:
        :param k_alpha:
        """
        max_val = 0
        for j in range(k_alpha):
            max_val += self.fun_support_decomp(vec_q, j)
        return max_val
