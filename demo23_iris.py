# demo23_iris   mulit classification

from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
data = iris.data
target = iris.target

regression1 = LogisticRegression(max_iter=200)
score = model_selection.cross_val_score(regression1, data, target, cv=3)
print(score)