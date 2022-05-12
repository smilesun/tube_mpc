import numpy as np


class ConstraintLNorminalNoKronecker():
    """ConstraintLNorminal.
    1. First row constraint:

    [-I_{d},[0]_d,[0]_d,  B_{d*r],[0]_{d*r],[0]_{d*r]] x = A_{d*d}[x_0]_{d*1}


    2. The rest equality constraint:

        Ax_k + Bu_k = x_{k+1}:

    Ax_1 + Bu_1 = x_2
    Ax_2 + Bu_2 = x_3
    [[A]_d,     -I_d,  [0]_{d*d} | [B]_{d*r}, [0]_{d*r}, [0]_{d*r} ] x = [0]_{d*1}
    [[0]_{d*d}, [A]_d, -I_d      | [0]_{d*r}, [B]_{d*r}, [0]_{d*r} ] x = [0]_{d*1}
    """

    def __init__(self):
        mat_sys = self.obj_dyn.mat_sys
        mat_input = self.obj_dyn.mat_input
        mat_sys = np.ones((d, d))
        n = 4  # horizon
        d = 2
        self = object()

    def __call__(self, x):
        """__call__.
        :param x: current state
        """
        d = x.shape[0]
        list_v_block = []
        block_first_row = self.build_first_line_constraint()
        list_v_block.append(block_first_row)
        for ind_row in range(1, d):
            block_row = self.build_eq_constraint_sys(ind_row)
            list_v_block.append(block_row)
        np.vstack(list_v_block)

    def build_first_line_constraint():
        """build_first_line_constraint."""
        # [[-I_{d},[0]_d,[0]_d,  B_{d*r],[0]_{d*r],[0]_{d*r]]
        block_zero = np.hstack([np.zeros((d, d)) for _ in range(d-1)])
        list_block = [-1 * np.eye(d), block_zero, self.mat_input, block_zero]
        np.hstack(list_block)


    def build_eq_constraint_sys(self, ind_row):
        """build_eq_constraint.
        :param ind_row:
        """
        block_middle = np.hstack([mat_sys, np.eye(d)])
        block_left, block_right = self.build_eq_constraint_sys_zero_left_right(ind_row, d)
        list_block = []
        if block_left:
            list_block.append(block_left)
        list_block.append(block_middle)
        if block_right:
            list_block.append(block_right)
        block_row = np.hstack(list_block)
        return block_row

    def build_eq_constraint_sys_zero_left_right(self, ind_row, d):
        """
        First row constraint:

        [-I_{d},[0]_d,[0]_d,  B_{d*r],[0]_{d*r],[0]_{d*r]] x = A_{d*d}[x_0]_{d*1}
        The rest equality constraint:

            Ax_k + Bu_k = x_{k+1}:

        Ax_1 + Bu_1 = x_2
        Ax_2 + Bu_2 = x_3
        [[A]_d,     -I_d,  [0]_{d*d} | [B]_{d*r}, [0]_{d*r}, [0]_{d*r} ]x = [0]_{d*1}
        [[0]_{d*d}, [A]_d, -I_d      | [0]_{d*r}, [B]_{d*r}, [0]_{d*r} ]x = [0]_{d*1}
        """
        block_zero_left = None
        block_zero_right = None
        if ind_row:  # ind_row start from 0 in python
            block_zero_left = np.hstack([np.zeros((d, d)) for _ in range(ind_row)])
        if ind_row & (ind_row < d-1):
            block_zero_right = np.hstack([np.zeros((d, d)) for _ in range(d-ind_row-1)])
        return block_zero_left, block_zero_right


    def build_eq_constraint_input(self, ind_row):
        """build_eq_constraint.
        :param ind_row:
        """
        mat_sys = self.obj_dyn.mat_sys
        mat_input = self.obj_dyn.mat_input
        d = 2
        np.eye(d)
        np.zeros((d, d))

    def build_eq_constraint_rhs(self, ind_row):
        """build_eq_constraint.
        :param ind_row:
        """
        mat_sys = self.obj_dyn.mat_sys
        mat_input = self.obj_dyn.mat_input
        d = 2
        np.eye(d)
        np.zeros((d, d))
