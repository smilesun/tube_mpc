"""
S_{\\infty}=minkowski_sum_{i=0:\\infty}(A+BK^{s})^i*W
S_{J}=minkowski_sum_{i=0:J-1}(A+BK^{s})^i*W
S_{J} \\subset  S_{\\infty} \\subset (1-\alpha)^{-1}S_{J}

(1-\alpha)^{-1}S_{J} provide a upper bound for S_{\\infty}, which
guarantees robustness, a upper bound for worst case disturbance
(when step wise disturbance w get aggegated till infinity)

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
S_{\\infty}= \\minkowski_sum_{i=0:\\infty}(A+BK^{s})^i*W

# decomposition of  S_{\\infty}
"""
import numpy as np
from tmpc.support_set_inclusion import is_implicit_subset_explicit


class ConstraintSAlpha():
    """
    S_{\\infty}=minkowski_sum_{i=0:\\infty}(A+BK^{s})^i*W
    S_{J}=minkowski_sum_{i=0:J-1}(A+BK^{s})^i*W
    S_{J} \\subset  S_{\\infty} \\subset (1-\alpha)^{-1}S_{J}

    # decomposition of  S_{\\infty}:
    define
    S_J= minkowski_sum_{i=0:J-1}{(A+BK^{s})^i*W}
    S_{\\infty} = minkowski_sum_{i=0:J-1}{(A+BK^{s})^i*W} + ...
    ... + minkowski_sum_{i=J:\\infty}{(A+BK^{s})^i*W}
    = S_J + minkowski_sum_{i=J:\\infty}{(A+BK^{s})^i*W}
    = S_J + minkowski_sum_{i=J:2J-1}{(A+BK^{s})^i*W} + ...
    ... + minkowski_sum_{i=2J:3J-1}{(A+BK^{s})^i*W} + ...
    ... + minkowski_sum_{i=3J:4J-1}{(A+BK^{s})^i*W} + ...
    # (note matrix  (A+BK^{s})^J can be extracted to the front,
    # but not the set W)
    = S_J + (A+BK^{s})^JS_J + (A+BK^{s})^JS_J + ...
    = S_J {(A+BK^{s})^0 + (A+BK^{s})^J  + (A+BK^{s})^{2J} + ...}

    # spectral radius inequality of (A+BK^{s})^J
    \\any \alpha \\in [0,1], \\exist integer J
    s.t. (A+BK^{s})^JW \\subset \alpha W
    # thus
    (A+BK^{s})^0(A+BK^{s})^JW \\subset \alpha W
    (A+BK^{s})^1(A+BK^{s})^JW \\subset \alpha (A+BK^{s})^1W
    (A+BK^{s})^2(A+BK^{s})^JW \\subset \alpha (A+BK^{s})^2W
    (A+BK^{s})^{J-1}(A+BK^{s})^JW \\subset \alpha (A+BK^{s})^{J-1}W
    # thus, sum over the above
    {(A+BK^{s})^0+ (A+BK^{s})^1 + ...+ (A+BK^{s})^{J-1}}(A+BK^{s})^JW ---
    ---\\subset \alpha {(A+BK^{s})^0+ (A+BK^{s})^1 + ...+ (A+BK^{s})^{J-1}}W
    <=>
    {(A+BK^{s})^0+ (A+BK^{s})^1 + ...+ (A+BK^{s})^{J-1}}(A+BK^{s})^JW  ---
    ---(A+BK^{s})^JS_J \\subset \alpha S_J
    # thus
    S_{\\infty}
    = S_J {(A+BK^{s})^0 + (A+BK^{s})^J  + (A+BK^{s})^{2J} + ...}
    \\subset S_J { \alpha^0 + \alpha + \alpha^2 + ...}
    =(1-\alpha)^{-1} S_J
    # Thus
    # S_J \\subset S_{\\infty} \\subset (1-\alpha)^{-1} S_J
    """
    def __init__(self, mat_sys, mat_input, mat_k_s, mat_w):
        """__init__."""
        self.mat_a_c = mat_sys + np.matmul(mat_input, mat_k_s)
        self.mat_w = mat_w
        self.s_alpha = None
        # pair of _alpha, _j_power
        self._alpha = None
        self._j_power = None
        # implicit representation of
        # S_J \\subset S_{\\infty} \\subset (1-\alpha)^{-1} S_J

    def cal_alpha_given_power(self, j_power):
        """cal_alpha_given_power.
        given J, calculate \alpha, s.t.
        (A+BK^{s})^JW \\subset \alpha W
        which result in:
        S_J \\subset S_{\\infty} \\subset (1-\alpha)^{-1} S_J
        """

    def cal_power_given_alpha(self, alpha):
        """cal_power_given_alpha."""

    def verify_inclusion(self, alpha, j_power):
        """
        How to verify:
        (A+BK^{s})^JW \\subset \alpha W, give $J$

        # solution 1:
            - W: {w| Mw<=1}
            -set \alpha W={\alpha*w|Mw<=1}
            = {w'=\alpha*w|M(\alpha)^{-1}(\alpha*w)<=1}
            = {w'|M(\alpha)^{-1}w'<=1}

            -set (A+BK^{s})^JW
            ={w'=(A+BK^{s})^Jw|Mw<=1}
            ={w'|Mw<=1 and w'=(A+BK^{s})^Jw}  # Note
            # (A+BK^{s})^J is not necessarily full rank, so
            # there is no explicit expression for this set
          solution 1 fails

        # solution 2:
            [(A+BK^{s})^J]W \\subset \alpha W, give $J$
            <=> \any w \\in {w|Mw<=1}
            \\exist  w', (A+BK^{s})^JW ={w'|M(\alpha)^{-1}w'<=1}
            s.t. A+BK^{s})^J*w = w'
            not sure how to achieve this?
            solution 2 fails

        # solution 3:
            # For two set

            Y={y=Ax|Mx<=1}  # in implicit form
            Z={x|Nx<=1}  # in explicit form
            to verify:  Y \\subset Z
            <=> {max}_x{N*A*x}=max_x{Ny|y=Ax, Mx<=1} <=1
            s.t. Mx<=1  (set {x|Mx<=1} defines Y={Ax})
            <=> if the maximum value of N*y: y=Ax & Mx<=1
            is smaller than 1, then y is inside Z.
            <=> h({Mx<=1}, N*A} <= 1

            # For the current case:
            [(A+BK^{s})^J]{W} \\subset \alpha {W} = {[M*(\alpha)^{-1}]*w'<=1}
            <=>
            max_{w} {[M*(\alpha)^{-1}](A+BK^{s})^J}w <=1
            s.t.
            M*w<=1
        """
        mat_a_set_y = np.linalg.matrix_power(self.mat_a_c, j_power)
        mat_x_set_y = self.mat_w
        mat_z_set_n = self.mat_w / alpha
        return is_implicit_subset_explicit(
            mat_a_set_y=mat_a_set_y,
            mat_x_set_y=mat_x_set_y,
            mat_z_set_n=mat_z_set_n)
