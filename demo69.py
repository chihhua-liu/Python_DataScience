# demo69  load image dataset and show image

import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load image dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# check image dataset
print(f"train image shape={train_images.shape},test image shape={test_images.shape}")
print(f"train label={len(train_labels)}, test label={len(test_labels)}")


def plotImage(index):  # show image
    plt.title(f"image marked as {train_labels[index]}")
    plt.imshow(train_images[index], cmap='binary')
    plt.show()


def plotTestImage(index): # show Image
    plt.title(f"test image marked as {test_labels[index]}")
    plt.imshow(test_images[index], cmap='binary')
    plt.show()


plotImage(0)
plotTestImage(2)