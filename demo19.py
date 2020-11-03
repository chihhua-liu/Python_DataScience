# demo19 sigmoid function

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
func1 = 1 / (1 + np.exp(-x))
plt.plot(x,func1)
plt.xlabel("observation")
plt.ylabel("probability (that will happen)")
plt.show()