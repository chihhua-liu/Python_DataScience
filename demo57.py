import numpy as np
from keras.layers import Dense
from keras.models import Sequential, save_model, load_model
from keras.callbacks import Callback
import time


# make a directory models

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
    print(m.summary())
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


MODEL_FILE = 'models/demo57'
model = createModel()

model.fit(inputList, resultList, epochs=200, batch_size=20, callbacks=[callback1])
save_model(model, MODEL_FILE)
score = model.evaluate(inputList, resultList)
print("[1]score=", score)
for s, n in zip(score, model.metrics_names):
    print("[1]{} value={}".format(n, s))

model2 = createModel()
score = model2.evaluate(inputList, resultList)
print("[2]score=", score)
for s, n in zip(score, model2.metrics_names):
    print("[2]{} value={}".format(n, s))

start = time.perf_counter()
model3 = load_model(MODEL_FILE)    # load_model
end = time.perf_counter()
print("[3]loading takes:{} seconds".format(end - start))
score = model3.evaluate(inputList, resultList)
print("[3]score=", score)
for s, n in zip(score, model3.metrics_names):
    print("[3]{} value={}".format(n, s))