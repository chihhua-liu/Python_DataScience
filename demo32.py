# demo32 K-Mean

from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 0], [0, 1], [1, 2], [1, 4], [1, 8],
              [2, 0], [4, 0], [4, 4], [4, 6], [4, 7]])
kmeans = KMeans(n_clusters=2).fit(X)
print("labels=", kmeans.labels_)
print("centers = ", kmeans.cluster_centers_)
print("predict as=", kmeans.predict([[0, 0], [4, 4], [-4, -4]]))
print("interia=",kmeans.inertia_)

