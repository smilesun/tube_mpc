import numpy as np
from p_i_s import iterate_invariance
from utils_plot_constraint import plot_polytope


def test_stable_A():
    A = np.array([[0.3, 0.5], [0.5, 0.3]])
    # A is stable, so one step reach invariant
    #  initial set is maximum
    M0 = np.array(
        [[0, 1],
        [1, 0],
        [-1, 0],
        [0, -1]])
    # constraint = iterate_invariance(mat0=M0, A=A, n_iter=10, call_back=plot_polytope)


def test_unstable_A():
    A = np.array([[1.1, 0.5], [0.5, 0.9]])
    M0 = np.array(
        [[0, 1],
        [1, 0],
        [-1, 0],
        [0, -1]])
    # constraint = iterate_invariance(mat0=M0, A=A, n_iter=5, call_back=plot_polytope)
