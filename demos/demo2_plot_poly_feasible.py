from intvalpy import  lineqs
import matplotlib.pyplot as plt
import numpy as np

A = -np.array([[5, 10],
                              [-1, 0],
                              [0, -1]])
b = -np.array([10, 0, 0])

lineqs(A, b, title='Solution', color='gray', alpha=0.5, s=10, size=(15,15), save=False, show=True)
plt.show()
