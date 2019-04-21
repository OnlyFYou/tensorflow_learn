import tensorflow as tf
'''
    指数衰减学习率
'''
global_step = tf.Variable(0)

learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize('myloss', global_step = global_step)