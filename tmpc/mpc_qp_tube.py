import numpy as np
import scipy
from tmpc.constraint_tightening import ConstraintTightening
from tmpc.constraint_tightening_z_terminal import ConstraintZT
from tmpc.constraint_eq_ldyn import ConstraintEqLdyn
from tmpc.constraint_block_horizon_lqr_solver import LqrQp
from tmpc.support_decomp import SupportDecomp
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
        v_0, v_1, .., v_{T-1}, z_1, z_2, ..., z_T, z_0, w_0, w_1, .., w{J-1}

    ## Matrix formulation:
    Finally, a matrix of the form
    M_{eq} [z_0^T, z_{1:T}^T, v_{0:T-1}^T, w_{1:J-1}^T]^T = b
    M_{ub} [z_0^T, z_{1:T}^T, v_{0:T-1}^T, w_{1:J-1}^T]^T <= [1]

    M_{eq} connect different decision variables
    M_{ub} will be in block diagonal form, each block does not have to be in
    the same dimension. Which means, each block can be constraints of the same
    applied to each stage variable, but different constraint can exsist.

    ## Problem definition:
        Cx+Du<=1
        C(z+s)+D(K^{z}*z+K^{s}s) <= 1
        (C+DK^{z})*z + (C+DK^{s})*s <= 1
        (C+DK^{z})*z + max_s{(C+DK^{s})*s} <= 1
        M^{z}*z + max_s {M^{s}*s} <=1 for t=0:\\infty

    ## constraint
        - input constraint: current state x_0=x=z_0+s_0
            -equality constraint for z_0 (different from nominal MPC):
                state variable s_0 can be decomposed of J step
                forward propagation
                define l=(1-\\alpha)^{-1}
                [I_z=A_{c, (s)}^0, [0]_{T*dim_sys+T*dim_input},...
                ...l*A_{c,(s)}^1, ..., l*A_{c,(s)}^{J-1}]
                [z_0^T, z_{1:T}, v_{0:T-1}, w_1^T, ..., w_{J-1}^T]^T = ...\\
                ...[0]_{dim_sys,1}
            - inequality constraint for z_0
                [[0]_{T*dim_sys+T*dim_input}, kron(mat_w, ones(1, J-1))]
                [z_0^T, z_{0:T}, v_{0:T-1}, w_1^T, ...,
                ...., w_{J-1}^T]^T <=[1]_{dim_sys, 1}

        - terminal constraint for z_T:
            - z_{T+1}=(A+BK^s)*z_T = A_{c, (s)}*z_T
            - stage constraint Mx_t<=1 (special case)
            - stage constraint Cx_t+Du_t<=1 (general case)
            - to ensure Mx_{T:\\infty} <=1, (special case) i.e.
            steps after T satisfies stage constraint
            <=>M(z_{t}+s_{t}) <=1, for t>T
            s.t. x_{t} = x_{t-1}+(K^z*z_{t-1}+K^s*s_{t-1}) + w_t
            - one step backward reachability:
                x_t = (A+BK^z)z_{t-1} + s_t
                x_t \\in X
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
            <=> M_{A_{c}=A+BK^z}z_{T:\\infty} <=1

        -equality constraint for nominal dynamic:
            Az_0 + Bv_0 = z_1
        ........................
        -constraint tightening for each stage
            original stage constraint for x_t (decoupled from u):
                Cx_{t}+Du_{t} <=1 (Cx_{t} <=1 (when D=0))
                <=>
                C(z_{t}+s_{t}) +D(K^{z}*z_{t}+K^{s}*s_t) <=1 (general form)
                <=>(C+DK^{z})*z + (C+DK^{s})*s <= 1
                <=>(C+DK^{z})*z + max_s{(C+DK^{s})*s} <= 1
                <=> M^{z}*z + max_s {M^{s}*s} <=1 for t=0:\\infty
                s.t. s \\in (1-\\alpha)^{-1}S_{J(\\alpha)}
                <=> for any i in nrow(M^{s})==nrow(M^{z}):
                    M^{z}[i, :]*z + max_s {M^{s}[i, :]*s} <=1
                s.t. s \\in (1-\\alpha)^{-1}S_{J(\\alpha)}
                <=> for any i in nrow(M^{s})==nrow(M^{z}):
                    M^{z}[i, :]*z + h(S_{J(\\alpha)}, M^{s}[i, :]^T) <=1
                    M^{z} = C+DK^{z}
                    M^{s} = C+DK^{s}


            #special case, when D=0
            now x_{t_k} = z_{t_k} + s_{t_k}
            Cx_{t_k} = C(z_{t_k} + s_{t_k}) <=1
            <=>Cz_{t_k} + max_{s_{t_k})}{Ms_{t_k})} <=1
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
                 mat_q, mat_r, mat_k_s,
                 mat_k_z,
                 mat_constraint4w,
                 constraint_x_u,
                 alpha_ini,
                 tolerance):
        """__init__.
        :param obj_dyn:
        """
        self.j_alpha = self.get_j_from_alpha(alpha_ini)
        self.builder_z_terminal = ConstraintZT(
            constraint_x_u,
            mat_constraint4w,
            mat_sys, mat_input,
            mat_k_s, mat_k_z,
            self.j_alpha,
            tolerance)

        self.mat_ub_block = None
        self.horizon = None
        self.stage_mat4z0 = self.builder_z_terminal.mat4z_terminal

        self.mat_constraint4z = constraint_x_u.mat_x + \
            np.matmul(constraint_x_u.mat_u, mat_k_z)
        self.mat_constraint4s = constraint_x_u.mat_x + \
            np.matmul(constraint_x_u.mat_u, mat_k_s)

        self.obj_support_decomp = SupportDecomp(
            mat_set=mat_constraint4w,
            mat_sys=mat_sys,
            mat_input=mat_input,
            mat_k_s=mat_k_s)

        self.j_alpha = self.get_j_from_alpha(alpha_ini)

        self.stage_mat4z = ConstraintTightening(
            self.mat_constraint4z,
            self.mat_constraint4s,
            self.obj_support_decomp,
            self.j_alpha)()

    def get_j_from_alpha(self, alpha_ini):
        return 4  # FIXME

    def build_mat_block_eq(self):
        """
        """

    def build_mat_block_ub(self, horizon, j_alpha):
        """
        M_{ub} [z_0^T, z_{1:T}^T, v_{0:T-1}^T, w_{1:J-1}^T]^T <= [1]
        M_{eq} connect different decision variables
        M_{ub} will be in block diagonal form, each block does not have to be
        in the same dimension. Which means, each block can be constraints of
        the same applied to each stage variable, but different constraint can
        exsist.
        """
        list_block_z = [self.stage_mat4z0]
        for _ in range(horizon):
            list_block_z.append(self.stage_mat4z)
        mat_left = scipy.linalg.block_diag(*list_block_z)
        mat_right = np.zeros((
            mat_left.shape[0],
            horizon*(self.dim_input+j_alpha)))
        self.mat_ub_block = np.hstack((mat_left, mat_right))
