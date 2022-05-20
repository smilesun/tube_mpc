import numpy as np
from tmpc.constraint_pos_inva_terminal import PosInvaTerminalSetBuilder
from tmpc.constraint_tightening import ConstraintTightening
from tmpc.support_decomp import SupportDecomp


class ConstraintZT():
    """
    - terminal constraint for z_T:
        - z_{T+1}=(A+BK^z)*z_T = A_c*z_T
        - stage constraint: Cx_t +Du_t<=1 (special case Cx_t<=1 when D=0)
            x_t = z_t + s_t
            u_t=K^{z}z_t + K^{s}s_t
            M^{z} = C+DK^{z}
            M^{s} = C+DK^{s}
            M^{z}z+M^{s}s<=1
        - to ensure Cx_{T:\\infty}+Du_{T:\\infty} <=1, i.e.
        steps after T satisfies stage constraint
        <=>M^{z}z_{t}+M^{s}s_{t} <=1, for t>T  (sharing the common M is special
        case when D=0 in Cx+Du<=1)
        ---
        s.t. x_{t} = Ax_{t-1}+B(K^{z}*z_{t-1}+K^{s}*s_{t-1}) + w_t
        decompose x_{t}=z_{t}+s_{t}
        s.t.
        &z_{t}
        = Az_{t-1} + Bv_{t-1}
        = Az_{t-1} + B{K^{z}z_{t-1}}
        = (A+BK^{z})z_{t-1}

        &s_{t}= x_{t} - z_{t}
        =Ax_{t-1}+B(K^{z}*z_{t-1}+K^{s}*s_{t-1}) + w_t - (A+BK^{z})z_{t-1}
        =A(z_{t-1}+s_{t-1})+B(K^{z}*z_{t-1}+K^{s}*s_{t-1}) + w_t ...
        ...- (A+BK^{z})z_{t-1}
        ={A+BK^{s}}s_{t-1} + w_t

        - one step backward reachability:
            x_t = (A+BK^{z})*z_{t-1} + (A+BK^{s})*s_{t-1} + **w_t**
            (do not forget the disturbance $w_t$)
            x_t, u_t \\in Y:={x,u|Cx+Du<=1}={x,
            u(x, K^z, K^s)|CX+D(K^z*z+K^s*s)<=1}
            <=>two simultaneous constraint
                -after 1 step, constraint still satisfied:
                    Cx_t+Du_t<=1
                    <=>
                    M^{z}z_t + M^{s}s_t <=1
                    <=>
                    M^z[(A+BK^z)z_{t-1}] + M^{s}[(A+BK^{s})s_{t-1}+w_{t-1}]<=1
                    where **s_t=(A+BK^{s})s_{t-1}+w_{t-1}**
                -current step feasible
                    M^{z}*z_{t-1} + M^{s}*s_{t-1} <=1

            <=>two simultaneous constraint
                -
                M^{z}[(A+BK^z)z_{t-1}]<= 1-max{M^{s}*[[A+BK^{s}]s_{t-1}+w_{t}]}
                **do not forget w_t**
                <=>
                M^{z}[(A+BK^z)z_{t-1}]<= 1-max{M^{s}*[s_t} ...
                ...<= 1-max{M^{s}*[s_{\\infty}} (note z_{t-1}, s_t of different
                time step occur in the same inequality, since we want to get
                rid of w_{t}, and since we pre-stabalize {s_t} sequence, the
                worst case $s$ can be used for the bound.
                -
                M^{z}*z_{t-1} <=1-max{M^{s}*s_{t-1}}<=1-max{M^{s}*s_{\\infty}}

        - convert to standard form of r.h.s. all ones
            define [1]=ones(nrow, 1) for below
            &([1]-max{M^{s}*s})^{-1}M^{z}[(A+BK^{z})z_{t-1}<= [1]
            &([1]-max{M^{s}*s})^{-1}M^{z}z_{t-1} <=[1]
            i.e.
            &(1-max{M^{s}*s})^{-1})[M^{z}(A+BK^{z})]z_{t-1}<= [1]
            &(1-max{M^{s}*s})^{-1})[M^{z}]z_{t-1} <=[1]
            i.e.
            &[[c_1]*M^{z}]A_c^{z}]z_{t-1}<= [1]
            (***[c_1]*M^{z} is row-wise multiplication***)
            &[[c_1]*M^{z}]z_{t-1} <=[1]
            i.e.
            & M'A_c^{z}z <= 1
            & M'z <= 1
            where,
            M'= [c_1]*M^{z} (***row-wise multiplication***),
            A_c^{z}= (A+BK^{z})
        - extend to infinity backward steps,i.e. starting from z_T
        nominal system should stay in a set which is positive
        invariant.
        <=> as long as
        M_{backward reachability}z_{T} <=1
        then
        M'z_{T:\\infinty} <= 1 (i.e. all subsequent steps until infinity
        will be constrained to stay in safe region)
        <=>((1-max{M^{s}*s})^{-1})[M^{z}]z_{T:\\infty} <=[1]
        under A_c^z = A+BK^{z}
    """
    def __init__(self, constraint_x_u,
                 mat_constraint4w,
                 mat_sys,
                 mat_input,
                 mat_k_s,
                 mat_k_z,
                 j_alpha,
                 tolerance,
                 max_iter4pos_inva=100
                 ):
        """__init__.
        :param constraint_x_u:
        :param mat_constraint4w:
        :param mat_sys:
        :param mat_input:
        :param mat_k_s:
        :param mat_k_z:
        :param tolerance:
        """
        self.max_iter4pos_inva = max_iter4pos_inva
        self.mat4z_terminal = None  # matrix to generate
        # Cx+Du<=1
        # C(z+s) + D(K^{z}z+K^{s}s<=1)
        # (C+DK^{z})z + (C+DK^{s})s<=1
        mat_constraint4z = constraint_x_u.mat_x + np.matmul(
            constraint_x_u.mat_u, mat_k_z)
        mat_constraint4s = constraint_x_u.mat_x + np.matmul(
            constraint_x_u.mat_u, mat_k_s)
        self.obj_support_decomp = SupportDecomp(
            mat_set=mat_constraint4w,
            mat_sys=mat_sys,
            mat_input=mat_input,
            mat_k_s=mat_k_s)

        self.mat_z = ConstraintTightening(
            mat_constraint4z=mat_constraint4z,
            mat_constraint4s=mat_constraint4s,
            obj_support_decomp=self.obj_support_decomp,
            j_alpha=j_alpha)()

        # A+BK^z
        """
            &[[c_1]*M^{z}]A_c^{z}]z_{t-1}<= [1]
            (***[c_1]*M^{z} is row-wise multiplication***)
            &[[c_1]*M^{z}]z_{t-1} <=[1]
            i.e.
            & M'A_c^{z}z <= 1
            & M'z <= 1
        A_c^{z} = A+BK^z does not need to be tightened
        """
        self.mat_a_c4z = mat_sys + np.matmul(mat_input, mat_k_z)
        self.pos_inva_builder = PosInvaTerminalSetBuilder(
            mat_sys=self.mat_a_c4z,  # no need to tighten
            mat_state_constraint=self.mat_z,  # already tightened
            tolerance=tolerance)

    def __call__(self):
        """
        """
        self.mat4z_terminal = self.pos_inva_builder(self.max_iter4pos_inva)
        return self.mat4z_terminal

    @property
    def constraint_mat(self):
        """constraint_mat."""
        return self.mat4z_terminal
