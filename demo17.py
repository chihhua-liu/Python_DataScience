#demo17'

from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import numpy as np

diabetes = datasets.load_diabetes()
print(type(diabetes), diabetes.data.shape, diabetes.target.shape)

dataForTest = -60
data_train = diabetes.data[:dataForTest]
print("data trained shape:", data_train.shape)

target_train = diabetes.target[:dataForTest]
print("target trained shape:", target_train.shape)

data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
print("data test shape:", data_test.shape)
print("target test shape:", target_test.shape)


regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)

print("score={}".format(regression1.score(data_test, target_test)))

for i in range(dataForTest, 0):
    dataArray = np.array(data_test[i]).reshape(1, -1)
    print("predict={:.2f}, actual={}".format(regression1.predict(dataArray)[0], target_test[i]))
mean_square_error = np.mean((regression1.predict(data_test) - target_test) ** 2)
print("MSE={}".format(mean_square_error))