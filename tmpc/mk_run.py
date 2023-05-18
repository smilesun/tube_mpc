from tmpc.simulate import Exp
from tmpc.dyn_sys import DynSysL


def mk_run(prob, controller, x_init, steps):
    dyn = DynSysL(dim_sys=prob.dim_sys,
                  dim_u=prob.dim_input,
                  x_ini=x_init,
                  constraint_x_u=controller.constraint_x_u,
                  max_w=prob.max_w,
                  mat_sys=prob.mat_sys,
                  mat_input=prob.mat_input)
    exp = Exp(dyn, controller=controller)
    exp.run(steps, controller.horizon)
