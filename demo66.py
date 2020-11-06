# demo66

import tensorflow as tf

vectors = [3.0, -1.0, 2.4, 5.9, 0.001, 8.5, -0.00000000001]
result1 = tf.nn.relu(vectors)    # nn is nu
result2 = tf.nn.sigmoid(vectors)
print(result1)
print(result2)