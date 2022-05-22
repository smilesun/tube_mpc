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
        mat_k = np.array([[-0.7384, -1.0998]])
        # return np.array([[1, 1]])
        return mat_k

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
        mat = np.array([[1.1, 1], [0, 1.3]])
        return mat

    @property
    def mat_input(self):
        mat = np.ones((2, 1))
        return mat

    @property
    def x_only_constraint(self):
        #mat_x = np.array([[2, -1],
        #                  [0, 0.2],
        #                  [0.2, 0],
        #                  [-0.2, 0],
        #                  [0, -0.2]])
        mat_x = np.array([[0, 0.2],
                          [0.2, 0],
                          [-0.2, 0],
                          [0, -0.2]])
        return mat_x

    @property
    def u_only_constraint(self):
        """
        |u|<1
        """
        mat_u = np.array([[0.5], [-0.5]])
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
        return 0.1
