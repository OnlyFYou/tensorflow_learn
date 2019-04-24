import tensorflow as tf

with tf.variable_scope('test'):
    a = tf.get_variable('a', [0], initializer=tf.constant_initializer(100.0))
    b = 0

print(a)
print(b)
