import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Mod import Model
batch_size = 100
# 指数衰减法基础学习率
learning_rate_base = 0.8
# 衰减率
learning_rate_decay = 0.99
# 正则化损失系数
reguzation_rate = 0.0001
# 总训练步数
train_steps = 10000
# 滑动平均衰减率
moving_average_decay = 0.99
model_path = '/model/to/model/'
model_log = '/model/to/path/log'


# 训练
def train(mnist):

    model = Model(model_path)
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')
    reguzation = tf.contrib.layers.l2_regularizer(reguzation_rate)
    hidden1 = model.nn_layer(x, 784, 500, None, reguzation, 'layer1')
    y = model.nn_layer(hidden1, 500, 10, None, reguzation, 'layer2')
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('moving_average'):
        avg_class = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        avg_op = avg_class.apply(tf.trainable_variables())
    with tf.name_scope('loss_function'):
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)))
        loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                                   mnist.train.num_examples/batch_size, learning_rate_decay)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        with tf.control_dependencies([train_step, avg_op]):
            train_op = tf.no_op(name='train')
    merge = tf.summary.merge_all()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 生成结构图信息
        writer = tf.summary.FileWriter(model_log, sess.graph)
        for i in range(train_steps):
            xs, ys = mnist.train.next_batch(batch_size)
            # 内存
            summary, loss_value, step, _ = sess.run([merge, loss, global_step, train_op], feed_dict={x: xs, y_: ys})
            writer.add_summary(summary, i)
            if i % 1000 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, loss_value, step, _ = sess.run([merge, loss, global_step, train_op], feed_dict={x: xs, y_: ys},
                                                        options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%03d' % i)
                print('after training step %s 步' % i)
        writer.close()
        model.save_model(sess, x, y)


def main():
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
