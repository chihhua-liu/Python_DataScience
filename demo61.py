# demo61'   KerasClassifier
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


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


def createModel():
    m = Sequential()
    m.add(Dense(14, input_dim=8, activation='relu'))
    m.add(Dense(8, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(m.summary())
    return m


model = KerasClassifier(build_fn=createModel, epochs=200, batch_size=20, verbose=1 )
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
result = cross_val_score(model, inputList, resultList, cv=fiveFold)
print("mean=%.3f,std=%.3f" % (result.mean(), result.std()))