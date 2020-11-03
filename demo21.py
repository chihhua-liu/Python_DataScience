#demo21

import matplotlib.pyplot as plt
import numpy as np

weights = [0.5, 1.0, 2.0, 3.0, 4.0]
title = "w=%d"

x = np.arange(-10, 10, 0.1)
for w in weights:
    f = 1 / (1 + np.exp(-w * x))
    plt.plot(x, f, label=title % w)
plt.legend(loc=2)
plt.show()