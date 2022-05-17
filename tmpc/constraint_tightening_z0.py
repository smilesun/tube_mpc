"""
$z_0$ constraint:
    .. math:: x=x_0 \\in X_0= {{z_0} \\minkowski_sum S}
    <=> x-z_0 \\in S=\\minkowski_sum_{i=0:k^{\alpha}-1}(A+BK^{s})^iW
    <=> x-z_0 = \\sum_{i=0:k^{\alpha}-1}(A+BK^{s})^i*w_i
              = \\sum_{i=0:k^{\alpha}-1}A_c^i*w_i
    s.t. w_i \\in W
    <=>
    x=z_0+\\sum_{i=0:k^{\alpha}-1}A_c^i*w_i (eq)
    mat_w * w_i <=1 (ub)
    <=> Let k^{alpha} = J
[I_z=A_c^0, A_c^1, ..., A_c^J]      [z_0^T, w_1^T, ..., w_{J}^T]^T = [0]_z
[[0]_z, kron(mat_w, ones(1, J))][z_0^T, w_1^T, ..., w_{J}^T]^T <=[1]
"""

class ConstraintZ0():
    def __init__(self, mat_a_c, mat_w, dim_sys, k_alpha):
        self.mat_a_c = mat_a_c
        self.mat_w = mat_w
        self.dim_sys = dim_sys
        self.k_alpha = k_alpha

    def __call__(self):
        list_a_c = []
        for i in range(self.k_alpha):
        list_a_c.append()


"""
###
S_k= \\minkowski_sum_{i=0:k-1}(A+BK^{s})^i*W
to judge if s_k \\ in S_k
(Note S_k is pre-stabalized, by choosing K^{s}, so A+BK^{s} is hurwitz)
<=>
there exist w_{0:k-1}, s.t.
- s_k = \sum_{i=0:k-1}(A+BK^{s})^i*w_i, equality constraint
- w_i \in W  or mat_disturbance * w_i <= 1

# state tubes:
X_0={z_0}+S, X_1={z_1}+S, ..., X_T={z_T}+S  # minkowski sum

# control tubes:
{v_0} + K^{s}*S,  {v_1} + K^{s}*S, ..., {v_T} + K^{s}*S,
#
#x_k = z_k + s_k \\in X_k \\ in X
#so s_k = x_k - z_k where z_k is the decision variable
#
#s_k = x_k - z_k
#s_{k} = (A+BK^{s})s_{k-1} + w_{k-1}
"""
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
