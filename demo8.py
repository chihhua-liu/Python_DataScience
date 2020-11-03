# demo8   sklearn and matplotrlib

import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

regressionData = datasets.make_regression(100, 1, noise=0)
print(type(regressionData))
features = regressionData[0]

print(features.min(), features.max())

labels = regressionData[1]
print(type(features), features.shape)
print(type(labels), labels.shape)
plt.scatter(features, labels, c='red', marker='.')
plt.show()
regression1 = linear_model.LinearRegression()
regression1.fit(regressionData[0], regressionData[1])
print("coef={}, intercept={}".format(regression1.coef_[0], regression1.intercept_))
range1 = [-3, 3]
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='blue')
plt.scatter(features, labels, c='red', marker='.')
plt.show()
print(regression1.score(features, labels))