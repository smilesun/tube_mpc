import numpy as np
from tmpc.constraint_pos_inva_terminal import iterate_invariance
from tmpc.utils_plot_constraint import plot_polytope


def test_stable_A():
    A = np.array([[0.3, 0.5], [0.5, 0.3]])
    # A is stable, so one step reach invariant
    #  initial set is maximum
    M0 = np.array(
        [[0, 1],
        [1, 0],
        [-1, 0],
        [0, -1]])
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=10)


def test_unstable_A():
    A = np.array([[1.1, 0.5], [0.5, 0.9]])
    M0 = np.array(
        [[0, 1],
        [1, 0],
        [-1, 0],
        [0, -1]])
    constraint = iterate_invariance(mat0=M0, A=A, n_iter=5)
