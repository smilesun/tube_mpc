import scipy
import numpy as np


def cal_k_discrete_lqr(mat_sys, mat_input, mat_loss_q, mat_loss_r):
    """Solve the discrete time lqr controller.
    x[k+1] = mat_sys x[k] + mat_input u[k]
    cost = sum x[k].T*mat_loss_q*x[k] + u[k].T*mat_loss_r*u[k]
    ---
    The DARE is defined as
     .. math::
    A^HXA - X - (A^HXB) (R + B^HXB)^{-1} (B^HXA) + Q = 0

    """
    # https://en.wikipedia.org/wiki/Algebraic_Riccati_equation

    mat_x = np.array(scipy.linalg.solve_discrete_are(
        mat_sys, mat_input, mat_loss_q, mat_loss_r))

    mat_k = scipy.linalg.inv(mat_input.T*mat_x*mat_input+mat_loss_r)
    mat_k = mat_k*(mat_input.T*mat_x*mat_sys)
    mat_k = np.array(mat_k)

    eigs, _ = scipy.linalg.eig(mat_sys-mat_input*mat_k)

    return mat_k, mat_x, eigs


def cal_k_continuous_lqr(mat_sys, mat_input, mat_loss_q, mat_loss_r):
    """Solve the continuous time lqr controller.
    dx/dt = mat_sys x + mat_input u
    cost = integral x.T*mat_loss_q*x + u.T*mat_loss_r*u
    """
    mat_x = np.array(
        scipy.linalg.solve_continuous_are(
            mat_sys, mat_input, mat_loss_q, mat_loss_r))

    mat_k = np.array(scipy.linalg.inv(mat_loss_r)*(mat_input.T*mat_x))

    eigs, _ = scipy.linalg.eig(mat_sys-mat_input*mat_k)
    return mat_k, mat_x, eigs
