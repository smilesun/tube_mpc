"""
S_{J(\\alpha)}= (1-\\alpha)^{-1}\\minkowski_sum_{j=0:J-1}(A+BK^{s})^j*{W}
(Note s_t, as state variable (sufficient statistic) for the disturbed system,
x_t = z_t + s_t,
is governed by pre-stabalized set dynamic on {S_t}, by choosing K^{s},
so A+BK^{s} is hurwitz)
to judge if s_0 \\ in S_J (positive invariant set of state variable)
(s_0 is initial state variable)
<=>
there exist disturbance w^{0:J-1} (note this is not the same time index, but
just forward propagation), s.t.
- s_0 = \\sum_{j=0:J-1}(A+BK^{s})^j*w^j, equality constraint
- w^j \\in W  or mat_w * w^j <= 1

# state tubes:
X_0={z_0}+S, X_1={z_1}+S, ..., X_T={z_T}+S  # minkowski sum
z_t = Az_{t-1}+Bv_t = Az_{t-1}+B*K^{z}*z_t = (A+BK^{z})*z_t

# control tubes:
{v_0} + K^{s}*S,  {v_1} + K^{s}*S, ..., {v_T} + K^{s}*S,
<=>{K^{z}z_0} + K^{s}*S,  {K^{z}z_1} + K^{s}*S, ..., {K^{z}z_T} + K^{s}*S,


# x_{t} = z_{t} + s_{t} \\in X_t \\ in X
so s_t = x_t - z_t where z_t is the decision variable

# backward decomp
s_{t} = (A+BK^{s})s_{t-1} + w_{t-1}
"""
import numpy as np
import scipy


def is_in_minkowski_sum(mat_set_1, mat_set_2):
    """
    x \\in set_1 + set_2
    <=>
    exist x_1 + x_2 = x, s.t. x_1 \\in set_1, x_2\\in set_2
    decision variable x_1, x_2
    constraint:
        x_1 + x_2 = x   # equality constraint
        mat_set_1 x_1 <= 1  # upper bound
        mat_set_2 x_2 <= 1  # upper bound
    ###
    """


class ConstraintZ0w():
    """
    $z_0$ constraint:
        x=x_0 \\in X_0= {{z_0} \\(1-\\alpha)^{-1}minkowski_sum S_{\\alpha}}
        <=> x-z_0 \\in (1-\\alpha)^{-1}S_{\\alpha}, where:
        S_{\\alpha}=(1-\\alpha)^{-1}\\minkowski_sum_{i=0:J_{\\alpha}-1}(A+BK^{s})^i*{W}
        <=> x-z_0 = (1-\\alpha)^{-1}\\sum_{i=0:J_{\\alpha}-1}(A+BK^{s})^i*w_i
                  = (1-\\alpha)^{-1}\\sum_{i=0:J_{\\alpha}-1}A_c^i*w_i
        s.t. w_i \\in W
        <=>
        x=z_0+(1-\\alpha)^{-1}\\sum_{i=0:J_{\\alpha}-1}A_c^i*w_i (eq)
        mat_w * w_i <=1 (ub)
        <=> Let J_{alpha} = J, l=(1-\\alpha)^{-1}

    - equality constraint:
    [I_z=A_c^0, l*A_c^1, ..., l*A_c^J][z_0^T, w_{1}^T, ..., w_{J}^T]^T = [0]_z

    - inequality constraint:
    [[0]_z, kron(mat_w, ones(1, J))][z_0^T, w_1^T, ..., w_{J}^T]^T <=[1]
    """
    def __init__(self, mat_a_c4s, mat_w, dim_sys, dim_input, j_alpha, alpha):
        """__init__.
        :param mat_a_c4s:
        :param mat_w:
        :param dim_sys:
        :param dim_input:
        :param j_alpha:
        :param alpha:
        """
        self.mat_a_c4s = mat_a_c4s
        self.mat_w = mat_w
        self.dim_sys = dim_sys
        self.dim_input = dim_input
        self._j_alpha = j_alpha
        self._alpha = alpha
        assert self._alpha < 1
        assert self._alpha > 0
        self._magnify = 1/(1-self._alpha)
        #
        self.mat_eq = None
        self.b_eq = None
        self.mat_ub = None
        self.horizon = None
        #
        assert self.mat_w.shape[1] == self.dim_sys

    def build_block_equality_constraint(self, horizon):
        """
        [I_z=A_c^0, [0]_{dim_sys, dim_sys*horizon,
        dim_input*horizon},
        l*A_c^1, ..., l*A_c^J][z_0^T, z_{1:T}, v_{0:T-1}
        w_1^T, ..., w_{J}^T]^T = [0]_z
        """
        if self.horizon == horizon:
            return self.mat_eq, self.b_eq
        mat_power = np.eye(self.mat_a_c4s.shape[0])
        list_a_c = [mat_power]
        mat_power *= self._magnify
        for _ in range(self._j_alpha):
            mat_power = np.matmul(mat_power, self.mat_a_c4s)
            list_a_c.append(mat_power)
        mat_eq4w = np.hstack(list_a_c)
        self.mat_eq = np.hstack(
            [np.eye(self.dim_sys),
             np.zeros((self.dim_sys, horizon*(self.dim_sys + self.dim_input))),
             mat_eq4w])
        self.b_eq = np.zeros((self.mat_eq.shape[0], 1))
        return self.mat_eq, self.b_eq

    def build_block_inequality_constraint4w(self, horizon):
        """
        horizon=4, dim_sys=2, dim_input=1
        [00,00,00,00]_z[0,0,0]_v[xx,00,00]_w
        [00,00,00,00]_z[0,0,0]_v[00,xx,00]_w
        [00,00,00,00]_z[0,0,0]_v[00,00,xx]_w
        ---
        Not the following!
        [[0]_{}, kron(mat_w, ones(1, J))]*
        [z_0^T, z_{1:T}, v_{0:T-1}, w_1^T, ..., w_{J}^T]^T <=[1]
        """
        mat_0 = np.zeros((
            self.dim_sys*self._j_alpha,
            (horizon+1)*self.dim_sys + horizon*self.dim_input))
        list_w = [self.mat_w for _ in range(self._j_alpha)]
        block_mat_w = scipy.linalg.block_diag(*list_w)
        self.mat_ub = np.hstack([mat_0, block_mat_w])
        return self.mat_ub
