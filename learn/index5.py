import tensorflow as tf


def interence(input_tensor, reuse=False):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable('weights', [784, 500], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [500], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable('weights', [500, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', 10, initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


x = tf.placeholder(tf.float32, [None, 784], name = 'x-input')
y = interence(x)

new_x = ...
new_y = interence(new_x, True)