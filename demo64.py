# demo64  for IMDB database search Data

import numpy
from keras.datasets import imdb
from matplotlib import pyplot as plt

# First Check  data from IMDBdatabase
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print(X_train.shape, X_test.shape, X.shape)
print(y_train.shape, y_test.shape, y.shape)
print(numpy.unique(y, return_counts=True))
print(len(numpy.unique(numpy.hstack(X))))
result = [len(x) for x in X]
print("-----------------------------------")
print(result)
print("-----------------------------------")
print("mean=%.2f, std=%.2f" % (numpy.mean(result), numpy.std(result)))

plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)
plt.show()