#demo2

import matplotlib.pyplot as plt
import numpy as np

b = 5
a = 3
x = np.arange(-10, 10, 0.1)
print(len(x), x)
y = a * x + b
plt.plot(x, y, label=f"y={a}x+{b}")
plt.legend(loc=2)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title("simple y=ax+b")
plt.xlabel('feature')
plt.ylabel('label')
plt.show()