# demo59

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split


class EarlyStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.5):
            print("\n**loss < 0.5, can stop**\n")
            self.model.stop_training = True


callback1 = EarlyStopCallback()

DATA_FILE = 'data/diabetes.csv'
dataset1 = np.loadtxt(DATA_FILE, delimiter=',', skiprows=1)
print(dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
# train_test_split ------------------------------------------------------------------
feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList, test_size=0.2,
                                                                        stratify=resultList)
for data in [resultList, label_train, label_test]:
    classes, counts = np.unique(data, return_counts=True)
    for cl, co in zip(classes, counts):
        print(f"{int(cl)}==>{co / sum(counts)}")

#print("--------------------------------")
model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(feature_train, label_train, epochs=200, batch_size=20, callbacks=[callback1],
          validation_data=(feature_test, label_test))
score = model.evaluate(feature_test, label_test)
print("score=", score)
for s, n in zip(score, model.metrics_names):
    print("{} value={}".format(n, s))