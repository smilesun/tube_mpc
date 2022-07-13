import numpy as np
from tmpc.support_fun import fun_support


class SupportDecomp():
    """
    # Support function on Minkowski sum of sets equals sum of support function
    - set $$S={minkowski_sum}_{j=0:J-1}S_j$$
      then for given arbitrary direction $$q$$
      $$h(S, q):= max_s {q^Ts|s=\\sum_{j=0:J-1}s_j \\in S}
      = max_s {q^T\\sum_{j=0:J-1}s_j |s_j\\in S_j}
      =\\sum_{j=0:J-1}max_{s_j} {q^Ts_j |s_j \\in S_j}
      =\\sum_{j=0:J-1}h(S_j, q)$$

      in summary:
          $$h(S, q) = \\sum_{j=0:J-1}h(S_j, q)$$

    - set $$S_j=A^j*{W}:={Aw | w \\in W}$$
      then
      $$
      h(S_j, q)
      = h(A^j*{W}, q) = max_{x\\in A^j*{W}}{q^Tx, s.t. x=A^jw, s.t. w \\in W}
      = max_{w\\in W} q^T(A^jw) = max_{w\\in W} ((A^j)^Tq)^Tw
      = h(W, q'=(A^j)^Tq)
      $$
      in summary:
      $$h(A^j*{W}, q) = h({W}, q'=(A^j)^Tq)$$

    ## Use Case:
    Once finite sequence **over** approximation of S_{\\infty}
    using spectral radius is found, conditioned on enlargement
    parameter $$(1-\\alpha)^{-1}$$, s.t.

    $$S_{J_{\\alpha}} \\subset S_{\\infty} \\subset  ...
    ...(1-\\alpha)^{-1}S_{J_{\\alpha}}$$

    where,
    $$S_{J_{\\alpha}}=\\minkowski_sum_{j=0:J_{\\alpha}-1}(A+BK_s)^j*{W}$$
    is an **under-approximation**, (however
    $$1/(1-\\alpha)S_{J_{\\alpha}}$$ is an **over-approximation**.

    and  $$J_{\\alpha}$$ satisfies

    *********************************************************************
    $$(A+BK_s)^{J_{\\alpha}}{W} \\subset \\alpha{W}$$
    *********************************************************************

    ## find Decomposition of support function w.r.t. $$S_{J_{\\alpha}}$$
    - $$h(S_{J_{\\alpha}}, q)=
    \\sum_{j=0:J_{\\alpha}-1}h(W, [(A+BK^{s})^j]^T*q)$$

    - $$h((1-\\alpha)^{-1}S_{J_{\\alpha}}, q)= ...
    ...\\sum_{j=0:J_{\\alpha}-1}(1-\\alpha)^{-1}h(W, [(A+BK^{s})^j]^T*q)$$
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
        self._mat_a_c_power = np.eye(self.mat_a_c.shape[0])

    def _support_power_transpose(self, vec_q, j_power):
        """fun_support_decomp.
        h(A^j*{W}, q) = h({W}, q'=(A^j)^Tq)
        :param vec_q:
        :param j_power:
        """
        mat_a_c_j = np.linalg.matrix_power(self.mat_a_c, j_power)  # FIXME: is iterative matrix multiplication faster?
        vec_direction = np.matmul(mat_a_c_j.T, vec_q)
        return fun_support(self.mat_set, vec_direction, b_ub=None)

    def decomp_support_minkow_sum(self, vec_q, j_alpha):
        """fun_support_minkow_sum.
        h(S, q) = \\sum_{j=0:J-1}h(S_j, q)
        :param vec_q:
        :param j_alpha:
        """
        max_val = 0
        for j in range(j_alpha):
            max_val += self._support_power_transpose(vec_q, j)
        return max_val
