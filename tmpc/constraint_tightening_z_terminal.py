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
        s.t. x_{t} = x_{t-1}+(K^{z}*z_{t-1}+K^{s}*s_{t-1})
        - one step backward reachability:
            x_t = (A+BK^{z})*z_{t-1} + (A+BK^{s})*s_{t-1}
            x_t, u_t \\in Y:={x,u|Cx+Du<=1}={x,
            u(x, K^z, K^s)|CX+D(K^z*z+K^s*s)<=1}
            <=>two simultaneous constraint
                -after 1 step, constraint still satisfied:
                    Cx_t+Du_t<=1
                    <=>
                    M^{z}z_t + M^{s}s_t <=1
                    <=>
                    M^z[(A+BK^z)z_{t-1}] + M^{s}[(A+BK^{s})s_{t-1}]<=1
                -current step feasible
                    M^{z}*z_{t-1} + M^{s}*s_{t-1} <=1
            <=>two simultaneous constraint
                -
                M^{z}[(A+BK^z)z_{t-1}]<= 1-max{M^{s}*[A+BK^{s}]s_{t-1}}
                -
                M^{z}*z_{t-1} <=1-max{M^{s}*s_{t-1}}<=1-max{M^{s}*s}

        - convert to standard form of r.h.s. all ones
            &(1-max{M^{s}*[A+BK^{s}]*s})^{-1}M^{z}[(A+BK^{z})z_{t-1}<= 1
            &(1-max{M^{s}*s})^{-1}M^{z}z_{t-1} <=1
            i.e.
            &(1-max{M^{s}*[A+BK^{s}]*s})^{-1}[M^{z}(A+BK^{z})]z_{t-1}<= 1
            &(1-max{M^{s}*s})^{-1}[M^{z}]z_{t-1} <=1
            i.e.
            &[c_1*M^{z}A_c^{z}]z_{t-1}<= 1
            &[c_2*M^{z}]z_{t-1} <=1
            i.e.
            &[c_2*M^{z}\frac{c_1}{c_2}A_c^{z}]z_{t-1}<= 1
            &[c_2*M^{z}]z_{t-1} <=1
            i.e.
            & M'A'z <= 1
            & M'z <= 1
            where,
            M'= c_2*M^{z},
            A'= \frac{c_1}{c_2}A_c^{z} = \frac{c_1}{c_2}(A+BK^{z})
        - extend to infinity steps
        <=> M_{backward-\\infty, A_c=A+BK^z, max{M*s}}z_{T} <=1
    """
    def __init__(self, mat_m4x, mat_m4w,
                 mat_sys,
                 mat_input,
                 mat_k_s,
                 mat_k_z,
                 tolerance
                 ):
        self.mat4z_terminal = None  # matrix to generate
        self.obj_support_decomp = SupportDecomp(
            mat_set=mat_m4w,
            mat_sys=mat_sys,
            mat_input=mat_input,
            mat_k_s=mat_k_s)

        self.mat_z = ConstraintTightening(
            mat_m4x=mat_m4x,
            obj_support_decomp=self.obj_support_decomp)

        # A+BK^z
        self.mat_a_c4z = mat_sys + np.matmul(mat_input, mat_k_z)
        self.pos_inva_builder = PosInvaTerminalSetBuilder(
            mat_sys=self.mat_a_c4z,
            mat_state_constraint=self.mat_z,
            tolerance=tolerance)
        ##

    def __call__(self):
        """
        """
        self.mat4z_terminal = self.pos_inva_builder()
        return self.mat4z_terminal
