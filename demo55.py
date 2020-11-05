# demo55
import tensorflow as tf
import keras
import numpy as np


def steak_model(x_new):
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    ys = np.array([500, 1000, 1500, 2000, 2500, 3000], dtype=float)
    layers = [keras.layers.Dense(units=1, input_shape=[1])]

    model = keras.Sequential(layers)
    model.compile(optimizer='sgd', loss="mean_squared_error")
    model.fit(xs, ys, epochs=5000)

    print(model.summary())
    return model.predict(x_new)[0]


print(steak_model([6]))