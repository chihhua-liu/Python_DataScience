#demo5

import matplotlib.pyplot as plt
import numpy as np

range1 = [-1, 3]
p = np.array([3])
plt.plot(range1, p * range1 + 5, 'ro-')
plt.show()
