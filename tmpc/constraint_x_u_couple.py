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
                 mat_x=None, mat_u=None, mat_xu_couple=None):
        """__init__.

        :param dim_sys:
        :param dim_input:
        :param mat_x:
        :param mat_u:
        :param mat_xu_couple:
        """
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
        if mat_xu_couple is not None:
            assert (dim_input + dim_sys) == mat_xu_couple.shape[1]
            mat_stack_xu = np.vstack([mat_stack_xu, mat_xu_couple])
        self._mat = mat_stack_xu
        self._mat_only_x = mat_x
        self._mat_only_u = mat_u

    @property
    def mat_only_x(self):
        return self._mat_only_x

    @property
    def mat_only_u(self):
        return self._mat_only_u

    @property
    def mat(self):
        """mat."""
        return self._mat

    @property
    def mat_xu_x(self):
        """mat_xu_x.
        C^T x + D^Tu <=1
        [C^T, D^T] u <=1
        return C^T
        """
        return self._mat[:, :self.dim_sys]

    @property
    def mat_xu_u(self):
        """mat_xu_u.
        C^T x + D^Tu <=1
        [C^T, D^T] u <=1
        return D^T
        """
        return self._mat[:, self.dim_sys:]

    def reduce2x(self, mat_k):
        """
        C^T x + D^Tu <=1
        C^T x + D^TKx <=1
        (C^T + D^TK)x <=1
        """
        return self.mat_xu_x + np.matmul(self.mat_xu_u, mat_k)

    def verify_x(self, vec_x):
        """verify_x.
        C^T x + D^Tu <=1
        [C^T, D^T] u <=1
        return c^Tx<=1
        :param vec_x:
        """
        if self._mat_only_x is None:
            return True
        assert all(np.matmul(self._mat_only_x, vec_x) < 1)
        return True  # if could pass assert

    def verify_u(self, vec_u):
        """verify_u.
        :param vec_u:
        """
        if self._mat_only_u is None:
            return True
        assert all(np.matmul(self._mat_only_u, vec_u) < 1)
        return True  # if could pass assert
