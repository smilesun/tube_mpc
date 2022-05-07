import numpy as np
import matplotlib.pyplot as plt
from intvalpy import lineqs

def plot_polytope(A_ub, b_ub=None):
    """plot_polytope.
    :param A_ub: A_ub*x <= b_ub
    """
    if b_ub is None:
        b_ub = -1*np.ones(A_ub.shape[0])
    ##
    ## lineqs: Ax>=b <==> -Ax<=-b
    ##
    list_vertices = lineqs(-1.0*np.array(A_ub), b_ub)  # Note that lineqs uses >= convention
    plt.show()
