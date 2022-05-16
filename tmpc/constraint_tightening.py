"""
# Let subscript denote time index
x_{k+1} = Ax_k+Bu_k+w_k
where
- w_k is disturbance, w_k \\in W where W is a bounded(compact)
disturbance set
- x_k \\in X_k  is the constraint of state


# Decompose the system into nominal part and disturbance part
x_k = z_k + s_k

## Note s_k != w_k, since w_k \\in W is bounded, but s_k \\ S_k can grow.

z_{k+1} = Az_k + Bv_k = Az_{k} + BK^{z}*z_{k} = (A+BK^{z})z_{k}
u_k = v_k + K^{s}*s_k

# Develop the dynamic of disturbance

x_{k+1} = Ax_k+Bu_k+w_k = Ax_k + B(K^{z}*z_k+K^{s}*s_k) + w_k
z_{k+1} = Az_k + Bv_k = Az_{k} + BK^{z}*z_{k} = (A+BK^{z})z_{k}
s_{k+1} := x_{k+1}-z_{k+1} = (A+BK^{s})s_k + w_k # note is w_k, not w_{k+1}

# Extend this to minkowski sum of set: set dynamic
s_{k+1} := x_{k+1}-z_{k+1} = (A+BK^{s})s_k + w_k
=>
S_{k+1} = (A+BK^{s})S_k + W   # minkowski sum of set, note disturbance set W is constant, like M0 for x

# Develop the recursive dynamic of disturbance
S_{k+1} = (A+BK^{s})S_k + W
= (A+BK^{s}) [(A+BK^{s})S_{k-1} + W] + W
= (A+BK^{s})^2*S_{k-1} +  (A+BK^{s})W + (A+BK^{s})^0 W

define S_{0}= W
S_k= \minkowski_sum_{i=0:k-1}(A+BK^{s})^i*W

If one can calculate S_{\infty}, then we know the worse case disturbance
x_k = z_k + s_k, if know the worst s_k, then we know how to constraint z_k
from the set perspective,
X_k = Z_k + S_k  # minkowski sum
we need a minkowski substraction
Z_k = X_k - S_k \\in X_k - S_{\infty} = {Z_k}^{worst}

# How to define {Z_k}^{worst}?
X_k = {x: x=z_k+s_k, z_k \\in Z_k, s_k \\ in S_{\infty}}
x \\in X_k for each time step $k$ <=> Mx_k<=1
<=>M(z_k+s_k)<=1
<=>Mz_k+Ms_k<=1
<=>for each i \\in nrows(M):
    M[i,:]*z_k+M[i,:]*s_k<=1
<=>
M[i,:]*z_k+ max_{s_k}[M[i,:]*s_k]<=1
<=>
M[i,:]*z_k+ h(S_k, M[i,:])<=1
where
h(S_k, M[i,:]):= max_{s_k \\in S_k} {M[i,:]s_k}
# finally
ensuring x_k \\in X_k is transformed to
M[i,:]*z_k+ h(S_k, M[i,:])<=1
<=>M[i,:]*z_k <=1-h(S_{k}, M[i,:])<=1-h(S_{\infty}, M[i,:])
"""


def fun_support(mat_set, vec_q):
    pass


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
    """


class ConstraintTightening():
    """
    S_k= \minkowski_sum_{i=0:k-1}(A+BK^{s})^i*W
    to judge if s_k \\ in S_k
    <=>
    there exist w_{0:k-1}, s.t.
    - s_k = \sum_{i=0:k-1}(A+BK^{s})^i*w_i, equality constraint
    - w_i \in W  or mat_disturbance * w_i <= 1
    #
    x_k = z_k + s_k \\in X_k \\ in X
    so s_k = x_k - z_k where z_k is the decision variable
    #
    s_k = x_k - z_k
    s_{k} = (A+BK^{s})s_{k-1} + w_{k-1}
    """
    def __init__(self, mat_m4x, mat_disturbance):
        self.mat_m4x = mat_m4x
        self.mat_disturbance = mat_disturbance   # w_k \\in W
