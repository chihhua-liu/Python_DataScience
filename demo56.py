# import numpy as np
# from keras.layers import Dense
# from keras.models import Sequential
#
# DATA_FILE = 'data/diabetes.csv'
# dataset1 = np.loadtxt(DATA_FILE, delimiter=',', skiprows=1)
# print(dataset1)
# print(dataset1.shape)
#
# inputList = dataset1[:, 0:8]
# resultList = dataset1[:, 8]
#
# model = Sequential()
# model.add(Dense(14, input_dim=8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# print(model.summary()) #141
#-------------------------------------
#demo56
#
# import numpy as np
# from keras.layers import Dense
# from keras.models import Sequential
#
# DATA_FILE = 'data/diabetes.csv'
# dataset1 = np.loadtxt(DATA_FILE, delimiter=',', skiprows=1)
# print(dataset1.shape)
#
# inputList = dataset1[:, 0:8]
# resultList = dataset1[:, 8]
#
# model = Sequential()
# model.add(Dense(14, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# print(model.summary())
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(inputList, resultList, epochs=200, batch_size=20)
# score = model.evaluate(inputList, resultList)
# print("score=", score)
# for s, n in zip(score, model.metrics_names):   # one score vs one  model.metrics_names(loss and accuracy)
#     print("{} value={}".format(n, s))
#--------------------------------------
# Add keras.callbacks
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback

# callback function---------------------
class EarlyStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.5):
            print("\n**loss < 0.5, can stop**\n")
            self.model.stop_training = True


callback1 = EarlyStopCallback()
# load data -----------------------------
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
model.fit(inputList, resultList, epochs=200, batch_size=20, callbacks=[callback1])  #  callbacks=[callback1]
score = model.evaluate(inputList, resultList)
print("score=", score)
for s, n in zip(score, model.metrics_names):
    print("{} value={}".format(n, s))