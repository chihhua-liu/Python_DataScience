# demo53 @tf.function increase execution efficiency

import tensorflow as tf

@tf.function
def add(p, q):
    return tf.add(p, q)


print(add([3, 4, 5], [-1, 2, 3]).numpy())