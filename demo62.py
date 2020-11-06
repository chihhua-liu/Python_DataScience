# Demo62   GridSearchCV
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import Callback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV


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


def createModel(optimizer='adam', init='uniform'):
    m = Sequential()
    m.add(Dense(14, input_dim=8, kernel_initializer=init, activation='relu'))
    m.add(Dense(8, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(m.summary())
    return m

print("-------------------------------------------------")
model = KerasClassifier(build_fn=createModel, verbose=0)
optimizers = ['sgd', 'rmsprop', 'adam']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(inputList, resultList)
print("best:%f using %s" % (grid_result.best_score_, grid_result.best_params_))