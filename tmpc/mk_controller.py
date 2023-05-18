"""
upon user constructed scenario and control input matrix
"""
from tmpc.mpc_qp_tube import MPCqpTube
from tmpc.constraint_x_u_couple import ConstraintStageXU


def mk_controller(prob, mat_k_s, x_init, horizon, alpha_ini=0.01, tolerance=0.01):
    """
    x_init is crutial for building block matrix
    """
    constraint_x_u = ConstraintStageXU(
        dim_sys=prob.dim_sys,  # dim of system
        dim_input=prob.dim_input,  # dim of input
        mat_x=prob.x_only_constraint,   # constraint for state
        mat_u=prob.u_only_constraint)   # constraint for input

    mpctube = MPCqpTube(
        mat_sys=prob.mat_sys,  # system
        mat_input=prob.mat_input,  # control input matrix
        mat_q=prob.mat_q,  # loss for state
        mat_r=prob.mat_r,  # loss for input
        mat_k_s=mat_k_s,   # control gain to supress disturbance
        mat_k_z=mat_k_s,   # control gain for ideal state
        mat_constraint4w=prob.mat_w,  # box bound for disturbance
        constraint_x_u=constraint_x_u,   # constraint for state and input
        alpha_ini=alpha_ini,
        tolerance=tolerance)

    mpctube.build_mat_block_ub(horizon=horizon)

    assert mpctube.mat_ub_block.shape[1] == \
        horizon*(prob.dim_input + prob.dim_sys) + prob.dim_sys \
        + mpctube.j_alpha * prob.dim_sys
    mpctube.build_mat_block_eq(x=x_init, horizon=horizon)
    return mpctube
