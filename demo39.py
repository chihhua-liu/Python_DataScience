# demo39 naive bayes classifier used  GaussianNB

# import numpy as np
# from sklearn.naive_bayes import GaussianNB
#
# X = np.array([[-1, -1], [-2, -1], [-3, -2],
#               [1, 1], [2, 1], [3, 2]])
# Y = np.array([1, 1, 1, 2, 2, 2])
# clf = GaussianNB()
# clf.fit(X, Y)
# print(clf.predict([[-0.8, -1], [0.5, 0.5], [-0.5, 0.5], [0, 0]]))
#
# clf_pf = GaussianNB()
# clf_pf.partial_fit(X, Y, np.unique(Y))   #  can used  partial_fit (do fit several times
# print(clf_pf.predict([[-0.8, -1]]))
# clf_pf.partial_fit([[0, 0]], [1])
# clf_pf.partial_fit([[-0.7, -0.8]], [2])
# print(clf_pf.predict([[-0.8, -1]]))
#---------------------------------------
# partial_fit

import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2],
              [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1], [0.5, 0.5], [-0.5, 0.5], [0, 0]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
# print(clf_pf.predict([[-0.8, -1]]))
print(clf_pf.predict([[0, 0]]))
clf_pf.partial_fit([[0.5, 0.5]], [2])
print(clf_pf.predict([[0, 0]]))
#clf_pf.partial_fit([[0, 0]], [1])
# clf_pf.partial_fit([[-0.7, -0.9]], [2])
#print(clf_pf.predict([[-0.8, -1]]))
