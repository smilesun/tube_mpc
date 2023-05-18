import numpy as np

def mk_problem(mat_sys, mat_input, max_disturbance,
               mat_x_constraint, mat_u_constraint):

    class Scenario():
        @property
        def dim_sys(self):
            assert mat_sys.shape[0] == mat_sys.shape[1]
            return mat_sys.shape[0]

        @property
        def dim_input(self):
            return mat_input.shape[1]

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
            return mat_sys

        @property
        def mat_input(self):
            return mat_input

        @property
        def x_only_constraint(self):
            return mat_x_constraint

        @property
        def u_only_constraint(self):
            """
            |u|<1
            """
            return mat_u_constraint

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
            return max_disturbance
    return Scenario
