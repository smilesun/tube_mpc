import numpy as np
from tmpc.support_fun import fun_support


def is_implicit_subset_explicit(mat_a_set_y, mat_x_set_y, mat_n_set_explicit):
    """
    :param: mat_a_set_y:  A
    :param: mat_x_set_y:  M, s.t. Mx<=1
    is implicit set(Y={y=Ax|Mx<=1}) a subset of an explicit set
    (mat_n_set_explicit*x <=1):

    # For two set
    Y={y=Ax|Mx<=1}  # in implicit form
    Z={x|Nx<=1}  # in explicit form
    to verify:  Y \\subset Z
    i.e. for \\any y \\in Y, y \\in Z
    <=> {max}_x{N*A*x}=max_x{Ny|y=Ax, Mx<=1} <=1
    s.t. Mx<=1  (set {x|Mx<=1} defines Y={Ax})
    <=> if the maximum value of N*y: y=Ax & Mx<=1
    is smaller than 1, then y is inside Z.
    <=> h({Mx<=1}, N*A} <= 1
    note
    A.shape = [dim(x), dim(x)]
    M.shape = [m, dim(x)]
    N.shape = [n, dim(x)]
    (N*A).shape = [n, dim(x)]
    """
    mat_n_times_a = np.matmul(mat_n_set_explicit, mat_a_set_y)
    nrow = mat_n_times_a.shape[0]
    for i in range(nrow):
        max_val = fun_support(
            mat_poly_set=mat_x_set_y,
            vec_half_plane_le=mat_n_times_a[i, :],
            b_ub=None)
        if max_val > 1:
            return False
    return True
