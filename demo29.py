#demo29

# manually create a directory graph
from subprocess import check_call
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
col = ['red', 'green']
marker = ['o', 'd']
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=col[type], marker=marker[type])
    index += 1
plt.show()
classifier = tree.DecisionTreeClassifier()
classifier.fit(X, Y)
OUT_FILE = 'graph/demo29.dot'
# OUT_GRAPH = 'graph/demo29.png'
# OUT_TYPE = '-Tpng'
# OUT_GRAPH = 'graph/demo29.svg'
# OUT_TYPE = '-Tsvg'
OUT_GRAPH = 'graph/demo29.pdf'
OUT_TYPE = '-Tpdf'

export_graphviz(classifier, out_file=OUT_FILE,
                filled=True, rounded=True, special_characters=True)

check_call(['dot', OUT_TYPE, OUT_FILE, '-o', OUT_GRAPH])