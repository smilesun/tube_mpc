import numpy as np
from tmpc.constraint_pos_inva_terminal import PosInvaTerminalSetBuilder
from tmpc.constraint_tightening import ConstraintTightening
from tmpc.support_decomp import SupportDecomp


class ConstraintZT():
    """
    - terminal constraint for z_T:
        - z_{T+1}=(A+BK^s)*z_T = A_c*z_T
        - stage constraint Mx_t<=1.
        - to ensure Mx_{T:\\infty} <=1, i.e.
        steps after T satisfies stage constraint
        <=>M(z_{t}+s_{t}) <=1, for t>T
        s.t. x_{t} = x_{t-1}+(K^z*z_{t-1}+K^s*s_{t-1})
        - one step backward reachability:
            x_t = (A+BK^z)z_{t-1} + s_t \\in X
            <=>two simultaneous constraint
            &M[(A+BK^z)z_{t-1} + s_t]<=1
            &M[z_{t-1} + s_t] <=1
            <=>two simultaneous constraint
            &M[(A+BK^z)z_{t-1}<= 1-max{M*s_t}<=1-max{M*s}
            &Mz_{t-1} <=1-max{M*s_{t-1}}<=1-max{M*s}
        - convert to standard form of r.h.s. all ones
            &(1-max{M*s})^{-1}M[(A+BK^z)z_{t-1}<= 1
            &(1-max{M*s})^{-1}Mz_{t-1} <=1
            i.e.
            &(1-max{M*s})^{-1}[MA_c]z_{t-1}<= 1
            &(1-max{M*s})^{-1}[M]z_{t-1} <=1
            i.e.
            &[M'A_c]z_{t-1}<= 1
            &[M']z_{t-1} <=1
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
