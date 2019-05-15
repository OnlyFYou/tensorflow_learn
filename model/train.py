import tensorflow as tf
from Data import DataUtil
from Mod import Model
import numpy as np
from Util import Activations

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
def train(total_size):
    model = Model()
    x = tf.placeholder(tf.float32, [None, 14], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, 2], name='y_input')
    reguzation = tf.contrib.layers.l2_regularizer(reguzation_rate)
    hidden1 = model.nn_layer(x, 14, 100, None, None, 'layer1', act=tf.nn.relu)
    y = model.nn_layer(hidden1, 100, 2, None, None, 'layer2', act=None)
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('moving_average'):
        avg_class = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
        avg_op = avg_class.apply(tf.trainable_variables())
    with tf.name_scope('loss_function'):
        # cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-12, 1.0)))
        cross_entropy = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=y_)
        # loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
        # mse = Losses.mse(y, y_)
        loss = cross_entropy #+ tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                                total_size/batch_size, learning_rate_decay)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        with tf.control_dependencies([train_step, avg_op]):
            train_op = tf.no_op(name='train')
    merge = tf.summary.merge_all()
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        # 生成结构图信息
        writer = tf.summary.FileWriter(model_log, sess.graph)
        train_e, train_l = DataUtil.get_data(['C:\\Workspace\\tensorflow_learn\\model\\data\\train.csv'], 15, 52)

        example_batch, label_batch = DataUtil.get_data(['C:\\Workspace\\tensorflow_learn\\model\\data\\test.csv'], 15,
                                                       batch_size)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(train_steps):
            # print('%s 训练' % i)
            e_batch, l_batch = sess.run([example_batch, label_batch])
            e_exam = [[0. if i.decode() == 'nan' else float(i.decode()) for i in x] for x in np.asarray(e_batch)]
            l_exam = [1 if x.decode('utf-8') == '1.0' else 0 for x in l_batch]
            label = tf.one_hot(l_exam, 2).eval()
            summary, loss_value, step, _ = sess.run([merge, loss, global_step, train_op],
                                                    feed_dict={x: e_exam, y_: label})
            writer.add_summary(summary, i)
            if i % 100 == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, loss_value, step, _ = sess.run([merge, loss, global_step, train_op],
                                                        feed_dict={x: e_exam, y_: label},
                                                        options=run_options, run_metadata=run_metadata)
                # writer.add_run_metadata(run_metadata, 'step%03d' % i)
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                a, b = sess.run([train_e, train_l])
                exam_train = [[0. if i.decode() == 'nan' else float(i.decode()) for i in x] for x in np.asarray(a)]
                label_train = [1 if x.decode('utf-8') == '1.0' else 0 for x in b]
                label = tf.one_hot(label_train, 2).eval()

                accuracy_score = sess.run(accuracy, feed_dict={x: exam_train, y_: label})
                print('after training step %s 步，正确率 %f' % (i, accuracy_score))
        coord.request_stop()
        coord.join(threads)
        writer.close()
        model.save_model(sess, x, y)


if __name__ == '__main__':
    train(4885)
