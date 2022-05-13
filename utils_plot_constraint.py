import numpy as np
import matplotlib.pyplot as plt
from intvalpy import lineqs

def plot_polytope(A_ub, text="feasible", b_ub=None):
    """plot_polytope.
    :param A_ub: A_ub*x <= b_ub
    """
    if b_ub is None:
        b_ub = -1*np.ones(A_ub.shape[0])
    A_ub = -1.0*np.array(A_ub)
    ##
    ## lineqs: Ax>=b <==> -Ax<=-b
    ##
    list_vertices = lineqs(A_ub, b_ub, title=text)  # Note that lineqs uses >= convention
    plt.show()
    return list_vertices
