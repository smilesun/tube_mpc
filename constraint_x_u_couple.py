import numpy as np


class ConstraintStageXU():
    """
    The existence of this class is
    - to check if the user input constraint make sense by visualizing
    - for easy user input specification
    - to document the x and u constraint:
    C^T x + D^Tu <=1 can be several rows
    first row:
        c1x + 0u <=1 will be [c1_{nrow(c1)*n}, 0_{nrow(c1)*r] [x^T, u^T]^T <=1
    second row:
        0x + d2u <=1 will be [0_{nrow(d1)*n}, d2_{nrow(d1)*r] [x^T, u^T]^T <=1
    third row:
        c3x + d3u <=1 will be [0_{nrow(d1)*n}
    Summarizing the above constraint:
        M_y y <=1
        where
        M_y = [[c1, 0], [0, d2], [c3, d3]]
        y = [x, u]
    """
    def __init__(self, dim_sys, dim_input,
                 mat_x=None, mat_u=None, mat_xu=None):
        self.dim_sys = dim_sys
        self.dim_input = dim_input

        if mat_x is not None:
            assert dim_sys == mat_x.shape[1]
            mat_xu1 = np.hstack([mat_x, np.zeros((mat_x.shape[0], dim_input))])
        mat_stack_xu = mat_xu1
        if mat_u is not None:
            assert dim_input == mat_u.shape[1]
            mat_xu2 = np.hstack([np.zeros((mat_u.shape[0], dim_sys)), mat_u])
            mat_stack_xu = np.vstack([mat_stack_xu, mat_xu2])
        if mat_xu is not None:
            mat_stack_xu = np.vstack([mat_stack_xu, mat_xu])
        self._mat = mat_stack_xu

    @property
    def mat(self):
        return self._mat

    def reduce2x(self):
        #mat_x = self._mat[:, :self.dim_sys]
        #mat_u = self._mat[:, self.dim_sys:]
        np.vstack([])
