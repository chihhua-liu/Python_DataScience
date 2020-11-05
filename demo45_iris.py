#demo45 for PCA : have 3D cart can use mouse check
# iris 4 group Data --> change to 3 group Data : 4D reduction 3D

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn import datasets
# from sklearn.decomposition import PCA
#
# iris = datasets.load_iris()
#
# X = iris.data
# species = iris.target
#
# fig = plt.figure(1, figsize=(8, 8))
# ax = Axes3D(fig, elev=-150, azim=110)
#
# X_reduced = PCA(n_components=3).fit_transform(iris.data)
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
#            c=species, cmap=plt.cm.Paired)
# print(iris.data.shape)
# print(X_reduced.shape)
# plt.show()
#----------------------------------------------------
#demo45'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

X = iris.data
species = iris.target

fig = plt.figure(1, figsize=(8, 8))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
           c=species, cmap=plt.cm.Paired)
print(iris.data.shape)
print(X_reduced.shape)
ax.set_xlabel("1st eigenvector")
ax.set_ylabel("2nd eigenvector")
ax.set_zlabel("3rd eigenvector")
plt.show()
