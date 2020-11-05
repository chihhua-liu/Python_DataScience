# demo50
import tensorflow as tf
# when tf>2.0, need this to enable legacy mode
tf.compat.v1.disable_eager_execution()
t1 = tf.constant("Hello old Tensorflow")
print(type(t1))
print(t1)
session1 = tf.compat.v1.Session()
print(session1.run(t1))
session1.close()