# demo6   LinearRegression() must used list
# y = label ， x= feature

from sklearn import linear_model
import matplotlib.pyplot as plt

regression1 = linear_model.LinearRegression()
features = [[1], [2], [3]]
values = [1, 4, 30]
plt.scatter(features, values, c='green')  #Show points
#plt.show()
regression1.fit(features, values)   # ca
print('linear regression coef={}, intercept={}'.format(regression1.coef_,
                                                       regression1.intercept_))
range1 = [-1, 3]
plt.plot(range1, regression1.coef_*range1+regression1.intercept_, c='gray')
print('score=',regression1.score(features, values))
plt.show()