

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

    def run(self, n_iter, horizon):
        """run.

        :param n_iter:
        :param horizon:
        """
        print("initial position:", self.dyn.x)
        self.dyn.verify_x()
        for i in range(n_iter):
            print("iteration", i)
            print("position:")
            print(self.dyn.x)
            try:
                vec_u = self.controller(self.dyn.x, horizon)
            except Exception as ex:
                info = "controller error at iteration %d: at state %s" \
                    % (i, str(self.dyn.x))
                raise RuntimeError(info + str(ex)) from ex
            self.dyn.step(vec_u)
            print("action:")
            print(self.dyn.u)
