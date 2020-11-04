# # demo34   K-meam
#
# from copy import deepcopy
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans
#
# X = np.r_[np.random.randn(50, 2) + [2, 2],
#           np.random.randn(50, 2) + [0, -2],
#           np.random.randn(50, 2) + [-2, 2]]
# print(X.shape)
#
# k = 3
# kmeans = KMeans(n_clusters=k)
# kmeans.fit(X)
# print(kmeans.cluster_centers_)
# print(kmeans.inertia_)
# colors = ['c', 'm', 'y', 'k']
# markers = ["o", '^', 'd', '*']
# for i in range(k):
#     dataX = X[kmeans.labels_ == i]
#     plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
#     print(dataX.size)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#             marker='*', s=200, c='#0559FF')
# plt.show()
#-----------------------------------------------
# demo34'  check inertia for K mean is better or worse

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.r_[np.random.randn(500, 2) + [2, 2],
          np.random.randn(500, 2) + [0, -2],
          np.random.randn(500, 2) + [-2, 2]]
print(X.shape)

k = 3  # check: k=4
kmeans = KMeans(n_clusters=k, verbose=True, n_init=20,max_iter=100)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print("inertia=", kmeans.inertia_)
colors = ['c', 'm', 'y', 'k']
markers = ["o", '^', '*', 'd']
for i in range(k):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
    print(dataX.size)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='*', s=200, c='#0559FF')
plt.show()