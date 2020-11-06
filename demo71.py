# demo71   image data conver 1D vector and 正規

# import tensorflow as tf
# import numpy as np
# from keras import Sequential
# from keras.layers import Dense
# from keras.datasets import mnist
#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# FLATTEN_DIM = 28 * 28
# TRAINING_SIZE = len(train_images)
# TEST_SIZE = len(test_images)
#
# trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
# testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
# print(train_images.shape, trainImages.shape)
#
# print(train_images[0])
# trainImages = trainImages.astype(np.float32)
# testImages = testImages.astype(np.float32)
# print(trainImages[0])
# trainImages /= 255
# testImages /= 255
# print(trainImages[0])
#-------------------------------------------
#demo71

# import tensorflow as tf
# import numpy as np
# from keras import Sequential
# from keras.layers import Dense
# from keras.datasets import mnist
# from keras import utils
#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# FLATTEN_DIM = 28 * 28
# TRAINING_SIZE = len(train_images)
# TEST_SIZE = len(test_images)
#
# trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
# testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
# # print(train_images.shape, trainImages.shape)
#
# # print(train_images[0])
# trainImages = trainImages.astype(np.float32)
# testImages = testImages.astype(np.float32)
# # print(trainImages[0])
# trainImages /= 255
# testImages /= 255
# # print(trainImages[0])
# print(train_labels[:10])
# NUM_DIGITS = 10
# trainLabels = utils.to_categorical(train_labels, NUM_DIGITS)
# testLabels = utils.to_categorical(test_labels, NUM_DIGITS)
#
# model = Sequential()
# model.add(Dense(units=128, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
# model.add(Dense(units=10, activation=tf.nn.softmax))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# print(model.summary())
# model.fit(trainImages, trainLabels, epochs=5)
# ----------------------------------------------

# #demo71'
#
# import tensorflow as tf
# import numpy as np
# from keras import Sequential
# from keras.layers import Dense
# from keras.datasets import mnist
# from keras import utils
#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# FLATTEN_DIM = 28 * 28
# TRAINING_SIZE = len(train_images)
# TEST_SIZE = len(test_images)
#
# trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
# testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
# # print(train_images.shape, trainImages.shape)
#
# # print(train_images[0])
# trainImages = trainImages.astype(np.float32)
# testImages = testImages.astype(np.float32)
# # print(trainImages[0])
# trainImages /= 255
# testImages /= 255
# # print(trainImages[0])
# print(train_labels[:10])
# NUM_DIGITS = 10
# trainLabels = utils.to_categorical(train_labels, NUM_DIGITS)
# testLabels = utils.to_categorical(test_labels, NUM_DIGITS)
#
# model = Sequential()
# model.add(Dense(units=128, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
# model.add(Dense(units=10, activation=tf.nn.softmax))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# print(model.summary())
# model.fit(trainImages, trainLabels, epochs=5, validation_data=(testImages, testLabels))
# predictedLabels = model.predict_classes(testImages)
# print("result:", predictedLabels[:10])
# predictedProbs = model.predict_proba(testImages)
# print("result:", predictedProbs[:10])
# predicted = model.predict(testImages)
# print('result:', predicted[:10])
# loss, accuracy = model.evaluate(testImages, testLabels)
# print("test accuracy:%.4f" % accuracy)
#----------------------------------
#demo71'
# tensorboard --logdir=logs\demo71  Check
#--------------------------------------------------------
# demo71'
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras import utils
from keras.callbacks import Callback, TensorBoard

# callback ------------------------
class EarlyStopCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get('val_accuracy') > 0.99):
            print("validation accuracy reach 99% correctness")
            self.model.stop_training = True


cb1 = EarlyStopCallback()

# load dataset ------------------------------------------------
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
# print(train_images.shape, trainImages.shape)

# print(train_images[0])
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
# print(trainImages[0])
trainImages /= 255
testImages /= 255
# print(trainImages[0])
print(train_labels[:10])
NUM_DIGITS = 10
trainLabels = utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = utils.to_categorical(test_labels, NUM_DIGITS)

model = Sequential()
model.add(Dense(units=128, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
model.add(Dense(units=64, activation=tf.nn.relu))
model.add(Dense(units=10, activation=tf.nn.softmax))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())
cb2 = TensorBoard(log_dir="logs/demo71", histogram_freq=0, write_graph=True, write_images=True)
model.fit(trainImages, trainLabels, epochs=100, validation_data=(testImages, testLabels), callbacks=[cb1, cb2])
predictedLabels = model.predict_classes(testImages)
print("result:", predictedLabels[:10])
predictedProbs = model.predict_proba(testImages)
print("result:", predictedProbs[:10])
predicted = model.predict(testImages)
print('result:', predicted[:10])
loss, accuracy = model.evaluate(testImages, testLabels)
print("test accuracy:%.4f" % accuracy)
