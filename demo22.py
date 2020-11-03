# demo22     # sigmoid function and cost function
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(list(iris.keys()))     #feature
print(iris.feature_names)   # 3, 4th
print(iris.target_names)    # 2, 3rd
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
print(X)
print(y)

regression1 = LogisticRegression()  # sigmoid
regression1.fit(X, y)
print(regression1.coef_)        # is a of ax+b
print(regression1.intercept_)   # is b of ax+b