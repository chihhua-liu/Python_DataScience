# SVM ?: support Vector Machine
import numpy as np
from sklearn.svm import SVC

x = np.array([[-1, -1], [-2, -1], [-3, -3], [1, 1], [2, 1], [3, 3]])
y = np.array([1, 1, 1, 2, 2, 2])
classifier = SVC()
classifier.fit(x, y)
print("predict=", classifier.predict([[-0.8, -1], [4, 4]]))