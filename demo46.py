# demo46 for PCA brief description

# from numpy import array
# from sklearn.decomposition import PCA
#
# A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
# print(A)
# pca = PCA(2)
# pca.fit(A)
# print(pca.components_)
# print(pca.explained_variance_)
# B = pca.transform(A)
# print(B)
#-------------------------------------
#demo46'
from numpy import array
from sklearn.decomposition import PCA

A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)
pca = PCA(2)
pca.fit(A)
print("--------------------------")
print(pca.components_)
print(pca.explained_variance_)
print("--------------------------")
B = pca.transform(A)
print(B)
print("--------------------------")
pca2 = PCA(2)
C = pca2.fit_transform(A)
print(C)
print(pca2.components_)
print(pca.explained_variance_)
from joblib import dump, load
print("--------------------------")
dump(pca, "demo46.joblib")
pca3 = load("demo46.joblib")
print(pca3.components_)
print(pca3.explained_variance_)