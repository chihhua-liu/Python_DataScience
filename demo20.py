# demo20
import matplotlib.pyplot as plt
import numpy as np

w1 = 3.0
b1, b2, b3, b4= -8, 0, 8, 16

l1 = 'b=-8.0'  # print label
l2 = 'b=0'
l3 = 'b=8.0'
l4 = 'b=16.0'

x = np.arange(-10, 10, 0.1)
for b, l in [(b1, l1), (b2, l2), (b3, l3), (b4, l4)]:
    f = 1 / (1 + np.exp(-(x * w1 + b)))
    plt.plot(x, f, label=l)
plt.legend(loc=2)
plt.show()
