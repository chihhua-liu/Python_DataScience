# demo48
#
# from sklearn.datasets import load_iris
# from sklearn.decomposition import PCA
# import numpy as np
# from sklearn import svm
# import matplotlib.pyplot as plt
#
# iris = load_iris()
# pca = PCA(n_components=2)
# data = pca.fit(iris.data).transform(iris.data)
# print(data.shape)
# datamax = data.max(axis=0) + 1
# datamin = data.min(axis=0) - 1
# print(datamax)
# print(datamin)
# n = 2000
# X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
#                    np.linspace(datamin[1], datamax[1], n))
# svc = svm.SVC()
# svc.fit(data, iris.target)
# Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
# print(np.unique(Z))
# plt.contour(X, Y, Z.reshape(X.shape))
# #plt.show()
# for i, c in zip([0, 1, 2], ['r', 'g', 'b']):
#     d = data[iris.target == i]
#     plt.scatter(d[:, 0], d[:, 1], c=c)
# plt.show()
#~~~~~~~~~~~~~~~~~
#demo48'
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

iris = load_iris()
pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
print(data.shape)
datamax = data.max(axis=0) + 1
datamin = data.min(axis=0) - 1
print(datamax)
print(datamin)
n = 2000
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))
svc = svm.SVC()
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
print(np.unique(Z))
plt.contour(X, Y, Z.reshape(X.shape), levels=[0, 1], colors=['r', 'g'])
# plt.show()
for i, c in zip([0, 1, 2], ['r', 'g', 'b']):
    d = data[iris.target == i]
    plt.scatter(d[:, 0], d[:, 1], c=c)
plt.show()