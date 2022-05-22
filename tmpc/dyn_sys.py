import numpy as np


class DynSysL():
    """DynSysL: linear dynamic system"""

    def __init__(self, dim_sys, dim_u, x_ini, constraint_x_u, max_w):
        """__init__.
        :param dim_sys:
        :param dim_u:
        """
        self.dim_u = dim_u
        self.dim_sys = dim_sys
        self.max_w = max_w
        self._constraint_x_u = constraint_x_u
        self._mat_sys = np.random.rand(dim_sys, dim_sys)
        self._mat_input = np.random.rand(dim_sys, dim_u)
        self._x = x_ini
        self._u = None

    @property
    def mat_sys(self):
        """mat_sys."""
        return self._mat_sys

    @property
    def mat_input(self):
        """mat_input."""
        return self._mat_input

    @property
    def x(self):
        return self._x

    @property
    def u(self):
        return self._u

    def verify_x(self):
        self._constraint_x_u.verify_x(self.x)

    def verify_u(self, u):
        self._constraint_x_u.verify_u(u)

    def gen_disturb(self):
        vec_w = np.random.randn(self.dim_sys, 1)
        ind = (vec_w > self.max_w)
        vec_w[ind] = self.max_w
        ind = (vec_w < -1.0*self.max_w)
        vec_w[ind] = -1.0 * self.max_w
        return vec_w

    def step(self, vec_u):
        self._u = vec_u
        assert vec_u.shape[1] == 1
        self.verify_u(vec_u)
        self._x = np.matmul(self._mat_sys, self._x) + \
            np.matmul(self.mat_input, vec_u) + self.gen_disturb()
        self.verify_x()
