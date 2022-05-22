import numpy as np
import scipy
from tmpc.constraint_tightening import ConstraintTightening
from tmpc.constraint_tightening_z_terminal import ConstraintZT
from tmpc.constraint_tightening_z0w import ConstraintZ0w
from tmpc.constraint_eq_ldyn_1_terminal import ConstraintEqLdyn1T
from tmpc.support_decomp import SupportDecomp
from tmpc.mpc_qp import MPCqp
from tmpc.block_lqr_loss import LqrQpLoss
from tmpc.solver_quadprog import quadprog_solve_qp
from tmpc.constraint_s_inf import ConstraintSAlpha


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
                 mat_q, mat_r,
                 mat_k_s, mat_k_z,
                 mat_constraint4w,
                 constraint_x_u,
                 alpha_ini,
                 tolerance, max_iter=100):
        """__init__.
        :param obj_dyn:
        """
        self.mat_q = mat_q
        self.mat_r = mat_r
        self.mat_ub_block = None
        self.b_ub = None
        self.horizon = None

        self.dim_sys = mat_sys.shape[0]
        self.mat_input = mat_input
        self.mat_sys = mat_sys
        self.dim_input = mat_input.shape[1]
        self.qp_loss = LqrQpLoss(mat_q, mat_r)
        constraint_j_alpha = ConstraintSAlpha(
            mat_sys=self.mat_sys,
            mat_input=self.mat_input,
            mat_k_s=mat_k_s,
            mat_w=mat_constraint4w,
            max_iter=max_iter)
        j_alpha = constraint_j_alpha.cal_power_given_alpha(alpha_ini)
        self.j_alpha = j_alpha

        mat_a_c4s = mat_sys + np.matmul(mat_input, mat_k_s)
        self.builder_z_0w = ConstraintZ0w(
            mat_a_c4s=mat_a_c4s,
            mat_w=mat_constraint4w,
            dim_sys=self.dim_sys,
            dim_input=self.dim_input,
            j_alpha=j_alpha,
            alpha=alpha_ini)

        self.builder_z_terminal = ConstraintZT(
            constraint_x_u,
            mat_constraint4w,
            mat_sys, mat_input,
            mat_k_s, mat_k_z,
            self.j_alpha,
            tolerance)

        self.stage_mat4z_terminal = self.builder_z_terminal()

        self.mat_constraint4z = constraint_x_u.mat_xu_x + \
            np.matmul(constraint_x_u.mat_xu_u, mat_k_z)
        self.mat_constraint4s = constraint_x_u.mat_xu_x + \
            np.matmul(constraint_x_u.mat_xu_u, mat_k_s)

        self.obj_support_decomp = SupportDecomp(
            mat_set=mat_constraint4w,
            mat_sys=mat_sys,
            mat_input=mat_input,
            mat_k_s=mat_k_s)

        self.stage_mat4z = ConstraintTightening(
            self.mat_constraint4z,
            self.mat_constraint4s,
            self.obj_support_decomp,
            self.j_alpha)()

        self.builder_eq4zv = ConstraintEqLdyn1T(
            self.mat_input,
            self.mat_sys)

    def __call__(self, x, horizon):
        mat_ub, b_ub = self.build_mat_block_ub(
            horizon=horizon)
        mat_eq, b_eq = self.build_mat_block_eq(
            x=x, horizon=horizon)
        mat_loss = self.qp_loss.gen_loss(
            self.mat_q, self.mat_r, horizon, self.j_alpha)
        vec = quadprog_solve_qp(
            P=mat_loss,
            A_ub=mat_ub,
            b_ub=b_ub,
            A_eq=mat_eq,
            b_eq=b_eq)
        pos = (horizon + 1)*(self.dim_sys)
        vec_u = vec[pos:pos+self.dim_input]
        vec_u = vec_u.reshape((len(vec_u), 1))
        return vec_u

    def build_mat_block_eq(self, horizon, x):
        """
        z_1=A*z_0 + Bv_0
        z_2=A*z_1 + Bv_1
        ...
        z_T=A*z_{T-1}+Bv_{T-1}
        ---
        z_0,z_1,z_2,z_3|v_0,v_1,v_2|w_1...w_J
        [ A, -I, 00,00 |  B,  0,  0|00, 00]
        [00,  A, -I,00 |  0,  B,  0|00, 00]
        [00,  00, A,-I |  0,  0,  B|00, 00]
        """
        mat_eq4w_z0, b4w_z0 = \
            self.builder_z_0w.build_block_equality_constraint(horizon, x)
        # dynamic equality
        mat_eq4zv = self.builder_eq4zv(horizon)
        mat_eq4zv_full = np.hstack(
            [mat_eq4zv,
             np.zeros((mat_eq4zv.shape[0], self.dim_sys*self.j_alpha))])
        # end dynamic equality
        block_mat_eq = np.vstack([mat_eq4w_z0, mat_eq4zv_full])

        block_b_eq = np.vstack([b4w_z0,
                                np.zeros((mat_eq4zv_full.shape[0], 1))])

        return block_mat_eq, block_b_eq

    def build_mat_block_ub(self, horizon):
        """
        (M_{eq} connect different decision variables)
        M_{ub} [z_0^T, z_{1:T}^T, v_{0:T-1}^T, w_{0:J-1}^T]^T <= [1]
        M_{ub} will be in block diagonal form, each block does not have to be
        in the same dimension, but the number of columns of each block
        should sum up to (horizon+1)*dim_sys+horizon*dim_input+J*dim_sys


        take horizon=4, dim_sys=2, dim_input=1
        # inequality constraint for z
        [xx,00,00,00]_z[0,0,0]_v[00,00,00]_w
        [00,xx,00,00]_z[0,0,0]_v[00,00,00]_w
        [00,00,xx,00]_z[0,0,0]_v[00,00,00]_w
        [00,00,00,xx]_z[0,0,0]_v[00,00,00]_w
        # no inequality constraint for v
        # inequality constraint for w
        [00,00,00,00]_z[0,0,0]_v[xx,00,00]_w
        [00,00,00,00]_z[0,0,0]_v[00,xx,00]_w
        [00,00,00,00]_z[0,0,0]_v[00,00,xx]_w

        """
        list_block_z = []
        for _ in range(horizon):  # FIXME:constraint tightening the same for z_0?
            list_block_z.append(self.stage_mat4z)
        list_block_z.append(self.stage_mat4z_terminal)
        mat_left = scipy.linalg.block_diag(*list_block_z)
        mat_right = np.zeros((
            mat_left.shape[0],
            horizon*self.dim_input + self.dim_sys*self.j_alpha))
        mat_ub_block_zv = np.hstack((mat_left, mat_right))
        block_mat4w = self.builder_z_0w.build_block_inequality_constraint4w(
            horizon)
        self.mat_ub_block = np.vstack([mat_ub_block_zv, block_mat4w])
        self.b_ub = np.ones((self.mat_ub_block.shape[0], 1))
        return self.mat_ub_block, self.b_ub
