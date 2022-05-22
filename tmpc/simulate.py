from tmpc.dyn_sys import DynSysL

class Exp():
    def __init__(self, dyn, controller):
        self.dyn = dyn
        self.controller = controller

    def run(self, n_iter, horizon, j_alpha):
        print("initial position:", self.dyn.x)
        self.dyn.verify_x()
        for i in range(n_iter):
            print("iteration", i, "position: ", self.dyn.x)
            vec_u = self.controller(self.dyn.x, horizon, j_alpha)
            self.dyn.step(vec_u)
