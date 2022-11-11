"""
interaction between dynamic system and controller
"""
from tmpc.memo import MemoTraj

class Exp():
    """Exp.
    combines the un-forced dynamic system and controller to
    simulate closed loop behavior.
    """
    def __init__(self, dyn, controller):
        """__init__.
        :param dyn:
        :param controller:
        """
        self.dyn = dyn
        self.controller = controller
        self.memo_traj=MemoTraj(
            shape_state=self.dyn.dim_sys,
            shape_u=self.dyn.dim_u,
            name_x='x', name_u='u')

    def run(self, nsteps, horizon):
        """run.
        :param nsteps:
        :param horizon: optimization horizon
        """
        print("initial position:", self.dyn.x)
        self.dyn.verify_x()
        for i in range(nsteps):
            print("step", i)
            print("position:")
            print(self.dyn.x)
            try:
                vec_u = self.controller(self.dyn.x, horizon)
            except Exception as ex:
                info = "controller error at step %d: at state %s" \
                    % (i, str(self.dyn.x))
                raise RuntimeError(info + str(ex)) from ex
            self.dyn.step(vec_u)
            print("action:")
            print(self.dyn.u)
            self.memo_traj.log_x_u(self.dyn.x, self.dyn.u)
