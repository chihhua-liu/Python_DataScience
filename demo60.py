#demo60

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.model_selection import StratifiedKFold


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

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)    # StratifiedKFold
totalScores = []


def createModel():
    m = Sequential()
    m.add(Dense(14, input_dim=8, activation='relu'))
    m.add(Dense(8, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    #print(m.summary())
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


for train, test in fiveFold.split(inputList, resultList):
    model = createModel()
    model.fit(inputList[train], resultList[train],
              epochs=200, batch_size=20, callbacks=[callback1], verbose=0)
    score = model.evaluate(inputList[test], resultList[test])
    print("score=", score)
    totalScores.append(score[1] * 100)
    for s, n in zip(score, model.metrics_names):
        print("{} value={}".format(n, s))
print("total accuracy: mean={:.3f}, std={:.3f}".format(np.mean(totalScores), np.std(totalScores)))