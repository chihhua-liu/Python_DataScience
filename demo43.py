#demo43
#pip install seaborn

#demo43'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = np.array([iris.target_names[i] for i in iris.target])
sns.pairplot(df, hue='species')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target,
                                                    test_size=0.3, stratify=iris.target)
rf1 = RandomForestClassifier(n_estimators=100, oob_score=True)
rf1.fit(X_train, y_train)

predicted = rf1.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print("oob score={}".format(rf1.oob_score_))
print("accuracy={}".format(accuracy))

cm1 = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)
print(cm1)
sns.heatmap(cm1, annot=True)
plt.show()

#demo43

