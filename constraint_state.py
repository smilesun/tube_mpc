import numpy as np

def get_constraint_state():
    M0 = np.array(
            [[0, 1],
            [1, 0],
            [-1, 0],
            [0, -1]])
    return M0
