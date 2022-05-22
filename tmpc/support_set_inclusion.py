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


def fun_is_subset(mat_set1, mat_set2, tolerance):
    """
    mat_set1 \\in mat_set2
    i.e. S_1 \\in S_2
    <=> for \\any s \\in S_1, => s \\ in S_2
    <=>M_1*x<=1 => M_2*x<=1
    """
    for i in range(mat_set2.shape[0]):
        half_plane_le = mat_set2[i]
        if not is_set_in_half_plane(mat_set1, half_plane_le,
                                    tolerance=tolerance):
            return False
    return True


def is_set_in_half_plane(mat_poly_set, half_plane_le,
                         tolerance,
                         b_ub=None,
                         fun_criteria=lambda x: x < 1):
    """
    polyhedra set is specified by np.matmul(mat_poly_set, x) <= 1
    support function is defined:
        h(mat_poly_set, q) = max_{x} q^T*x, s.t. x \\in S={mat_poly_set x <=1}
    """
    # NOTE: should be mat_poly_set,
    # not K-step backward reachability constraint matrix!
    max_val = fun_support(mat_poly_set, half_plane_le, b_ub)

    if fun_criteria(max_val-tolerance):
        # if worst case satisfies constraint, then no need for this constraint
        # to exist
        return True
    return False
