import numpy as np
from tmpc.constraint_pos_inva_terminal import PosInvaTerminalSetBuilder
from tmpc.constraint_eq_ldyn import ConstraintEqLdyn
from tmpc.constraint_block_horizon_lqr_solver import LqrQp
from tmpc.mpc_qp import MPCqp


class MPCqpTube(MPCqp):
    """
    T: horizon of planning where T is the terminal point
    dim_s: dimension of dynamic system
    Dynamic (Equality) Constraint:
        [diag[A]_{T-1}, [0]_{T-1*dim_s}, diag[B]_{T-1}}]

        :dim(diag[A]_{T-1}) = (dim_s * (T-1)) * (dim_s * (T-1))
        :dim(diag[B]_{T-1}}) = (dim_s*(T-1)) * (dim_u * (T-1))
    suppose T = 3
    decision variables:
        v_0, v_1, v_2=v_{T-1}
        z_0, z_1, z_2, z_3=x_{T}
    input: current state x_0=x=z_0+s_0
    -equality constraint for z_0
    -inequality constraint for z_0
    -equality constraint for nominal dynamic:
        Az_0 + Bv_0 = z_1
    ........................
    -Inequality constraint: constraint tightening for each stage
    [0, 0, M_{inf_tightening}, 0, 0] [z_1, ..., z_T] < 1
    """
    def __init__(self, mat_sys, mat_input,
                 mat_q, mat_r, mat_k,
                 constraint_x_u):
        """__init__.
        :param obj_dyn:
        """
