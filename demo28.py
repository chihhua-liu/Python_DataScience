from sklearn import tree
from matplotlib import pyplot

X = [[0, 0], [1, 1]]

Y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)

nodes = [[2,2],[-1,1],[1,-1]]
print(clf.predict(nodes))

tree.plot_tree(clf)
pyplot.show()