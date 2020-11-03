#demo18

# import matplotlib.pyplot as plt
# from sklearn import datasets
#
# iris = datasets.load_iris()
#
# print(type(iris))
# print(dir(iris))
# # for item in dir(iris):
# #     print(iris[item])
# labels = iris['feature_names']
# print(labels)
# print("=================")
# X = iris.data
# species = iris.target
# counter = 1
# for i in range(4):
#     for j in range(i + 1, 4):
#         plt.figure(counter, figsize=(8, 6))
#         plt.scatter(X[:, i], X[:, j], c=species, cmap=plt.cm.Paired)
#         plt.show()
#------------------------------------------------------
#demo18'

import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

print(type(iris))
print(dir(iris))
# for item in dir(iris):
#     print(iris[item])
labels = iris['feature_names']  #sepal length,sepal width,petal length,petal width
labels1 =iris['target_names']   # 3 type flower
print(labels )
print(labels1)
#-----------------------------
X = iris.data
species = iris.target
counter = 1
for i in range(4):
    for j in range(i + 1, 4):
        plt.figure(counter, figsize=(8, 6))
        xData = X[:, i]
        yData = X[:, j]
        x_min, x_max = xData.min() - .5, xData.max() + .5
        y_min, y_max = yData.min() - .5, yData.max() + .5
        plt.scatter(xData, yData, c=species, cmap=plt.cm.Paired)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()
