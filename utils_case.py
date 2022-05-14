import numpy as np

A = np.matrix([[0.3, 0.5], [0.5, 0.3]])  # A is stable, so one step reach invariant
#  initial set is maximum

A = np.matrix([[1.1, 0.5], [0.5, 0.9]])
np.linalg.eig(A)

M0 = np.matrix([
    [0, 1],
    [1, 0],
    [-1, 0],
    [0, -1]])


class Probset():
    @property
    def mat_sys(self):
        mat = np.array([[1.1, 0.5], [0.5, 0.9]])
        return mat

    @property
    def mat_input(self):
        mat = np.zeros((2, 1))
        mat[1] = 0.8
        return mat

    @property
    def x_only_constraint(self):
        mat_x = np.array([[2, -1],
                           [0, 1],
                           [1, 0],
                           [0, -1]])
        return mat_x

    @property
    def dim_sys(self):
        return 2

    @property
    def dim_input(self):
        return 1
