#demo23

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))
print(iris.feature_names)  # 3, 4th
print(iris.target_names)  # 2, 3rd
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
print(X)
print(y)

regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_)
print(regression1.intercept_)
plt.plot(X, y, 'b.')

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = regression1.predict_proba(x_new)
y_predict = 1 / (1 + np.exp(-(regression1.coef_[0] * x_new + regression1.intercept_)))
plt.plot(x_new, y_prob[:, 1], 'g-', label='iris-verginica')
plt.plot(x_new, y_prob[:, 0], 'r--', label='not iris-verginica')
plt.plot(x_new, y_predict, 'b--', label="calculate")
plt.legend(loc='upper left')
plt.show()