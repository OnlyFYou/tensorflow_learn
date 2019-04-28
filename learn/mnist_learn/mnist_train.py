# _*_ coding: utf-8 _*_
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import learn.mnist_learn.mnist_inference

BATCH_SIZE = 100
# 指数衰减法基础学习率
LEARNING_RATE_BASE = 0.8
# 学习衰减率
LEARNING_RATE_DECAY = 0.99
# 正则化损失的模型复杂度
REGULARZATION_RATE = 0.0001
TRAINING_STEPS = 9000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = '/path/to/model'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, [None, learn.mnist_learn.mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, learn.mnist_learn.mnist_inference.OUTPUT_NODE], name='y-input')
    # 正则化类
    regularizer = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    y = learn.mnist_learn.mnist_inference.inference(x, regularizer)

    global_step = tf.Variable(0, trainable=False)
    # 滑动平均类
    variable_average = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 反向传播改变网络参数和滑动平均值
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                # 一千轮训练保存一次模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                print('训练 %s 千轮' % (i/1000))
        writer = tf.summary.FileWriter("/path/to/log", sess.graph)
    writer.close()


def main():
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
