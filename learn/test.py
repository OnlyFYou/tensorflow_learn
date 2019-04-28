import tensorflow as tf

with tf.variable_scope('foo'):
    a = tf.get_variable('bar', [1])
    print(a.name)