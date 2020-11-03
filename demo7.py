# demo7   3D data : y= ap+bq+c --> get plane， if up 3D get hyper plane

import matplotlib.pyplot as plt
# from sklearn import linear_model
#
# features = [[0, 1], [1, 3], [2, 6]]
# values = [1, 4, 5.5]
# regression = linear_model.LinearRegression() #LinearRegression()
#
# regression.fit(features, values)             #fit(features, values)
# print(regression.coef_)                      # coef_
# print(regression.intercept_)                 # intercept_
#
# f1 = [[0, 0], [1, 1], [2, 2], [3, 3]]        #predict()
# l1 = regression.predict(f1)                  #predict()
# print(l1)
# print(regression.score(features, values))
# print(regression.score(f1, l1))
# print(regression.score(f1, [3, 8, 12, 17]))

#=================================================
# demo7' modify
import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 6], [3, 7]]
values = [1, 4, 5.5, 6]
regression = linear_model.LinearRegression()
regression.fit(features, values)
print(regression.coef_)
print(regression.intercept_)
print("---------------------------------")
f1 = [[0, 0], [1, 1], [2, 2], [3, 3]]
l1 = regression.predict(f1)
print(l1)
estimate_value = regression.predict(features)       # 期望值
print("期望值---------------------------------")
print("estimate_value={}".format(estimate_value))
print("real value=", regression.score(features, values))
print("ideal value=", regression.score(features, estimate_value))
print("idea value=", regression.score(f1, l1))
print("idea value with offset=", regression.score(f1, [3, 8, 12, 17]))