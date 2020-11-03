# demo25'  PCA
#
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets
# from sklearn import svm
# from sklearn.decomposition import PCA
#
# iris = datasets.load_iris()
# pca = PCA(n_components=2)
#
# data = pca.fit(iris.data).transform(iris.data)
# print(iris.data.shape)
# print(data)
# print("------------------------------")
# print(data.shape)
# dataMax = data.max(axis=0) + 1
# dataMin = data.min(axis=0) - 1
# n = 2000
# X, Y = np.meshgrid(np.linspace(dataMin[0], dataMax[0], n),
#                    np.linspace(dataMin[1], dataMax[1], n))
# print(type(X), X.shape)
# print(type(Y), Y.shape)
# print("------------------------------")
# print(X)
# print(Y)
# print("------------------------------")
# svc = svm.SVC()
# svc.fit(data, iris.target)
# Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
#
# plt.contour(X, Y, Z.reshape(X.shape), colors='k')
# for c, s in zip([0, 1, 2], ['o', '.', '*']):
#     d = data[iris.target == c]
#     plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)
# print(svc.score(data,iris.target))
# plt.show()

#--------------------------------------
# demo25'
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.decomposition import PCA

iris = datasets.load_iris()
pca = PCA(n_components=2)

data = pca.fit(iris.data).transform(iris.data)
print(iris.data.shape)
print(data.shape)
dataMax = data.max(axis=0) + 1
dataMin = data.min(axis=0) - 1
n = 2000
X, Y = np.meshgrid(np.linspace(dataMin[0], dataMax[0], n),
                   np.linspace(dataMin[1], dataMax[1], n))
print(type(X), X.shape)
print(type(Y), Y.shape)
# kernel=linear|poly|rbf|sigmoid
# poly C=1, 0.94666
# poly C=10,0.96
# poly C =100,
svc = svm.SVC(kernel="linear",C=100)    # used : C
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
plt.contour(X, Y, Z.reshape(X.shape), colors='k')
for c, s in zip([0, 1, 2], ['o', '.', '*']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)
print(svc.score(data,iris.target))
plt.show()