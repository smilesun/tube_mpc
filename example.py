import numpy as np
from tmpc import mk_problem, mk_controller, mk_run


mat_sys = np.array([[1.1, 1], [0, 1.3]])

mat_input = np.ones((2, 1))

mat_x_constraint = np.array([[0, 0.2],
                             [0.2, 0],
                             [-0.2, 0],
                             [0, -0.2]])

mat_u_constraint = np.array([[0.5], [-0.5]])

max_disturbance = 0.1


prob = mk_problem(mat_sys, mat_input, max_disturbance,
                  mat_x_constraint,
                  mat_u_constraint)


mat_k = np.array([[-0.7384, -1.0998]])


x = np.array([[0.01, 0.01]]).T
controller = mk_controller(prob, mat_k, x, horizon=10)


controller(x, horizon=10)

mk_run(prob, controller, x, steps=30)
