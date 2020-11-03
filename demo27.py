#demo27_iris_2

from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

iris = datasets.load_iris()
data = iris.data
target = iris.target

regression1 = LogisticRegression(max_iter=200)
svc1 = SVC(kernel='poly')
svc2 = SVC(kernel='linear')
svc3 = SVC(kernel='rbf')
svc4 = SVC(kernel='sigmoid')
classifiers = [regression1, svc1, svc2, svc3, svc4]
for c in classifiers:
    score = model_selection.cross_val_score(c, data, target, cv=3)
    print("{} has score{}".format(c, score))