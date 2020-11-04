# # demo37'   Add :import : (1) from sklearn.model_selection import cross_val_score (2) import joblib
#
# # make a directory data
# # copy sonar-all to data
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split   # used train-test
#
# df = pd.read_csv('data/sonar.all-data', header=None, prefix='X')
# print(df.shape)
# data, labels = df.iloc[:, :-1], df.iloc[:, -1]
# print(data.shape)
# print(labels.shape)
# df.rename(columns={'X60': "Label"}, inplace=True)
# print(df.columns)
# clf = KNeighborsClassifier(n_neighbors=6)
#
# X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
# clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
# print("score=", clf.score(X_test, y_test))
#---------------------------------------------
# demo37'

# make a directory data
# copy sonar-all to data
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

df = pd.read_csv('data/sonar.all-data', header=None, prefix='X')
print(df.shape)
data, labels = df.iloc[:, :-1], df.iloc[:, -1]
print(data.shape)
print(labels.shape)
df.rename(columns={'X60': "Label"}, inplace=True)
print(df.columns)
clf = KNeighborsClassifier(n_neighbors=6)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print("score=", clf.score(X_test, y_test))
result_cm1 = confusion_matrix(y_test, y_predict)
print(result_cm1)

scores = cross_val_score(clf, data, labels, cv=3, groups=labels)
print(scores)

from joblib import dump, load

dump(clf,"knn1.joblib")
knn2 = load('knn1.joblib')
y_predict2 = knn2.predict(X_test)
result2 = confusion_matrix(y_predict, y_predict2)
print(result2)
