# demo41 naive_bayes

import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
#Y = np.array([1, 2, 2, 1, 2, 2])
Y = np.array([1, 1, 2, 1, 1, 2])
X_min, X_max = -4, 4
Y_min, Y_max = -4, 4

h = .025
xx, yy = np.meshgrid(np.arange(X_min, X_max, h), np.arange(Y_min, Y_max, h))
clf = GaussianNB()
clf.fit(X, Y)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.pcolormesh(xx, yy, Z)
plt.show()
