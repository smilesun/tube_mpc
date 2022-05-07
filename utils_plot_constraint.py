import numpy as np
import matplotlib.pyplot as plt
from intvalpy import lineqs

def plot_polytope(M0, b=None):
    """plot_polytope.
    :param M0: M0*x <= 1
        """
    if b is None:
        b = -1*np.ones(M0.shape[0])
    matrix = lineqs(np.array(M0), b, title='Solution', color='gray',
                    alpha=0.5, s=10, size=(15, 15), save=False,
                    show=True)  # Note that lineqs uses >= convention
    plt.show()
