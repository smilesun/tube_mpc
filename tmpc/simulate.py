

class Exp():
    def __init__(self, dyn, controller):
        self.dyn = dyn
        self.controller = controller

    def run(self, n_iter, horizon):
        print("initial position:", self.dyn.x)
        self.dyn.verify_x()
        for i in range(n_iter):
            print("iteration", i)
            print("position:")
            print(self.dyn.x)
            try:
                vec_u = self.controller(self.dyn.x, horizon)
            except Exception as e:
                info = "controller error at iteration %d: at state %s" % (i, str(self.dyn.x))
                raise RuntimeError(info + str(e))
            self.dyn.step(vec_u)
            print("action:")
            print(self.dyn.u)

