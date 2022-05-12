import numpy as np
from constraint_state import get_constraint_state


class DynSysL():
    """DynSysL: linear dynamic system"""

    def __init__(self, dim_sys, dim_u):
        """__init__.
        :param dim_sys:
        :param dim_u:
        """
        self.dim_u = dim_u
        self.dim_sys = dim_sys
        self._mat_sys = np.random.rand(dim_sys, dim_sys)
        self._mat_input = np.random.rand(dim_sys, dim_u)
        self._mat_state_constraint = get_constraint_state()

    @property
    def mat_sys(self):
        """mat_sys."""
        return self._mat_sys

    @property
    def mat_input(self):
        """mat_input."""
        return self._mat_input

    @property
    def mat_state_constraint(self):
        """mat_state_constraint."""
        return self._mat_state_constraint
