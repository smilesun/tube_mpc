"""
"""
import numpy as np


class ConstraintTightening():
    """
        -constraint tightening for each stage
            original stage constraint for x_t (decoupled from u):
                Cx_{t}+Du_{t} <=1
                Mx_{t} <=1 (when D=0)
                C(z_{t}+s_{t}) +D(K^{z}*z_{t}+K^{s}*s_t) <=1 (general form)
                <=>(C+DK^{z})*z + (C+DK^{s})*s <= 1
                <=>(C+DK^{z})*z + max_s{(C+DK^{s})*s} <= 1
                <=> M^{z}*z + max_s {M^{s}*s} <=1 for t=0:\\infty
                s.t. s \\in S_{J(\\alpha)}
                <=> for any i in nrow(M^{s})==nrow(M^{z}):
                    M^{z}[i, :]*z + max_s {M^{s}[i, :]*s} <=1
                s.t. s \\in S_{J(\\alpha)}
                <=> for any i in nrow(M^{s})==nrow(M^{z}):
                    M^{z}[i, :]*z + h(S_{J(\\alpha)}, M^{s}[i, :]^T) <=1
                    M^{z} = C+DK^{z}
                    M^{s} = C+DK^{s}


    M[i,:]*z_k <=1-h(S_{k}, M[i,:])<=1-h(S_{\\infty}, M[i,:])
    where S_{\\infty} is the worst case stage disturbance

# Disturbed system dynamic

Let subscript denote time index:
x_{k+1} = Ax_k+Bu_k+w_k
where,
- w_k is disturbance, w_k \\in W where W is a bounded(compact) disturbance set
- x_k \\in X_k  is the constraint of state


# Decompose the system into nominal part and disturbance part

x_k = z_k + s_k

!!!! Note s_k != w_k, since w_k \\in W is bounded, but s_k \\ S_k can grow, s_k
reflect aggregated effect of disturbance up to the current step. s_k is a state
variable

z_{k+1} = Az_k + Bv_k = Az_{k} + BK^{z}*z_{k} = (A+BK^{z})z_{k}
u_k = v_k + K^{s}*s_k    (note s_k is a state variable of aggregated effect of
disturbance)

# Develop the dynamic of disturbance

x_{k+1} = Ax_k+Bu_k+w_k = Ax_k + B(K^{z}*z_k+K^{s}*s_k) + w_k
z_{k+1} = Az_k + Bv_k = Az_{k} + BK^{z}*z_{k} = (A+BK^{z})z_{k}
s_{k+1} := x_{k+1}-z_{k+1} = (A+BK^{s})s_k + w_k # note is w_k, not w_{k+1}

# Extend this to minkowski sum of set: set dynamic
s_{k+1} := x_{k+1}-z_{k+1} = (A+BK^{s})s_k + w_k
=>
S_{k+1} = (A+BK^{s})S_k + W   # minkowski sum of set,
# note disturbance set W is constant, like M0 for x

# Develop the recursive dynamic of disturbance
S_{k+1} = (A+BK^{s})S_k + W
= (A+BK^{s}) [(A+BK^{s})S_{k-1} + W] + W
= (A+BK^{s})^2*S_{k-1} +  (A+BK^{s})W + (A+BK^{s})^0 W

define S_{0}= W
S_k= \\minkowski_sum_{i=0:k-1}(A+BK^{s})^i*W

If one can calculate S_{\\infty}, then we know the worse case disturbance
x_k = z_k + s_k, if we know the worst s_k, then we know how to constraint z_k
from the set perspective,
X_k = Z_k + S_k  # minkowski sum
we need a minkowski substraction
Z_k = X_k - S_k \\in X_k - S_{\\infty} = {Z_k}^{worst}

# How to define {Z_k}^{worst}?
Suppose the constraint for x_k is Mx_k<=1
X_k = {x: x=z_k+s_k, z_k \\in Z_k, s_k \\ in S_{\\infty}}
x \\in X_k for each time step $k$ <=> Mx_k<=1
<=>M(z_k+s_k)<=1
<=>Mz_k+Ms_k<=1
<=>for each i \\in nrows(M):
    M[i,:]*z_k+M[i,:]*s_k<=1
<=>
M[i,:]*z_k+ max_{s_k}[M[i,:]*s_k]<=1
<=>
M[i,:]*z_k+ h(S_k, M[i,:])<=1
where,
h(S_k, M[i,:]):= max_{s_k \\in S_k} {M[i,:]s_k}
finally,
ensuring x_k \\in X_k is transformed to
M[i,:]*z_k+ h(S_k, M[i,:])<=1
<=>M[i,:]*z_k <=1-h(S_{k}, M[i,:])<=1-h(S_{\\infty}, M[i,:])

# How to calculate h(S_{\\infty}, M[i,:])?

    """
    def __init__(self, mat_m4x, obj_support_decomp):
        """__init__.
        :param mat_m4x: stage wise constraint for x: mat_m4x*x<=1
        :param obj_support_decomp: object for support function decomposition
        """
        self.mat_m4x = mat_m4x
        self.k_alpha = 4   # FIXME
        self.obj_support_decomp = obj_support_decomp

    def _handle_ith_row(self, i):
        """_handle_ith_row.

        :param i:
        """
        vec_q = self.mat_m4x[i, :]
        h_support = self.obj_support_decomp.fun_support_minkow_sum(
            vec_q, self.k_alpha)
        return vec_q / (1-h_support)   # FIXME: check divide by zero?

    def __call__(self):
        """__call__."""
        list_mat4z = []
        for i in range(self.mat_m4x.shape[0]):
            row = self._handle_ith_row(i)
            list_mat4z.append(row)
        mat4z = np.vstack(list_mat4z)
        return mat4z
