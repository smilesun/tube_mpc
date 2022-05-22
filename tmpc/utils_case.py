import numpy as np
import control


class Probset():
    @property
    def x_init(self):
        x = np.array([[0.2, 0.2]]).T
        return x

    @property
    def dim_sys(self):
        return 2

    @property
    def dim_input(self):
        return 1

    @property
    def mat_k(self):
        mat_k_s = control.place(self.mat_sys, self.mat_input, [-0.2, 0.1])
        mat_k_s = -1.0 * mat_k_s
        # return np.array([[1, 1]])
        return mat_k_s

    @property
    def mat_q(self):
        mat = np.eye(2)
        return mat

    @property
    def mat_r(self):
        mat = np.eye(1)
        return mat

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
                          [-1, 0],
                          [0, -1]])
        return mat_x

    @property
    def u_only_constraint(self):
        """
        |u|<1
        """
        mat_u = np.array([[1], [-1]])
        return mat_u

    @property
    def mat_w(self):
        """
        100*w_1 <= 1
        -100*w_1 <= 1  <=> w_1 >=-0.01
        100*w_x <=1
        -100*w_2 <= 1  <=> w_2 >=-0.01
        """
        coefficient = 1/self.max_w
        mat_constraint4w = np.array([
            [coefficient, 0],
            [-1.0*coefficient, 0],
            [0, coefficient],
            [0, -1.0*coefficient]])
        return mat_constraint4w

    @property
    def max_w(self):
        return 0.01
