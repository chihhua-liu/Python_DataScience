#demo58 retraining

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback
import matplotlib.pyplot as plt


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

model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(inputList, resultList, epochs=200, validation_split=0.1,
          batch_size=20, callbacks=[callback1])

score = model.evaluate(inputList, resultList)
print("score=", score)
for s, n in zip(score, model.metrics_names):
    print("{} value={}".format(n, s))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(['accuracy','val_accuracy'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.show()