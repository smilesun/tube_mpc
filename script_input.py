import numpy as np

A = np.matrix([[0.3, 0.5], [0.5, 0.3]])  # A is stable, so one step reach invariant
#  initial set is maximum
A = np.matrix([[1.1, 0.5], [0.5, 0.9]])
np.linalg.eig(A)

M0 = np.matrix([
        [2, -1],
    [0, 1],
    [1, 0],
    [0, -1]])

M0 = np.matrix([
    [0, 1],
    [1, 0],
    [-1, 0],
    [0, -1]])



