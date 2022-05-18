import numpy as np
from tmpc.constraint_pos_inva_terminal import PosInvaTerminalSetBuilder
from tmpc.constraint_eq_ldyn import ConstraintEqLdyn
from tmpc.constraint_block_horizon_lqr_solver import LqrQp
from tmpc.mpc_qp import MPCqp


class MPCqpTube(MPCqp):
    """
    T: horizon of planning where T is the terminal point
    dim_s: dimension of dynamic system
    suppose T = 3
    decision variables:
        v_0, v_1, v_2=v_{T-1}
        z_0, z_1, z_2, z_3=z_{T}
        w_0, w_1, w_{J-1} auxiliary decision variable,
        does not affect objective
    decision variable in long vector form:
        v_0, v_1, .., v_{T-1}, z_1, z_2, ..., z_T, z_0, w_0, w_1, .., w{j-1}

    ## constraint
        - input constraint: current state x_0=x=z_0+s_0
            -equality constraint for z_0:
                state variable s_0 can be decomposed of J step
                forward propagation
                [I_z=A_c^0, l*A_c^1, ..., l*A_c^J]
                [z_0^T, w_1^T, ..., w_{J}^T]^T = [0]_z
            - inequality constraint for z_0
                [[0]_z, kron(mat_w, ones(1, J))]
                [z_0^T, w_1^T, ..., w_{J}^T]^T <=[1]

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

        -equality constraint for nominal dynamic:
            Az_0 + Bv_0 = z_1
        ........................
        -constraint tightening for each stage
            original stage constraint for x_t:
                Mx_{t_k} <=1
            now x_{t_k} = z_{t_k} + s_{t_k}
            Mx_{t_k} = M(z_{t_k} + s_{t_k}) <=1
            <=>Mz_{t_k} + max_{s_{t_k})}{Ms_{t_k})} <=1
            <=>
            M[i,:]*z_k <=1-h(S_{k}, M[i,:])<=1-h(S_{\\infty}, M[i,:])
            <=> {M[i,:]/[1-h(S_{\\infty}, M[i,:])]}*z_k <= 1
            let nrow(M_{inf_tightening})
            [np.zeros(nrow, dim_sys)},
            kron(M_{inf_tightening}, ones(1, T)),
            kron(zeros(1, T), ones(nrow, dim_sys)),
            kron(zeros(1, T), ones(nrow, dim_input))]
            [z_0, z_1, ..., z_T, w_{1:T}, v_{0:T-1}] < 1
    """
    def __init__(self, mat_sys, mat_input,
                 mat_q, mat_r, mat_k,
                 constraint_x_u):
        """__init__.
        :param obj_dyn:
        """
