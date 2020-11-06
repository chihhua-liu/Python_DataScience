#demo 63  1-hot encoding

from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras import utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

df = read_csv('data/iris.data', header=None)
dataset = df.values
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
# 1-hot encoding -------------------------------------------
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
dummy_y = utils.to_categorical(encoded_Y)
print(labels)
print(encoded_Y)
print(dummy_y)


def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=1)
kfold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features, dummy_y, cv=kfold)
print("acc: %.4f, std: %.4f" % (results.mean() * 100, results.std() * 100))