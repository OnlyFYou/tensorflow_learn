import tensorflow as tf
from Data import DataUtil
from Mod import Model
import numpy as np


class TrainModel:
    @staticmethod
    def train_start(data_info):
        train_file_ary = data_info['train_file_ary']
        test_file_ary = data_info['test_file_ary']
        total_steps = data_info['total_steps']
        model_path = data_info['model_path']
        model_log = data_info['model_log']
        n_node_ary = data_info['n_node_ary']
        feature_size = data_info['feature_size']
        act = data_info['act']
        regu_bool = data_info['regu_bool']
        loss_fun = data_info['loss_fun']
        train_data_size = data_info['train_data_size']
        batch_size = data_info['batch_size']
        test_data_size = data_info['test_data_size']

        # 指数衰减法基础学习率
        learning_rate_base = 0.8
        # 衰减率
        learning_rate_decay = 0.99
        # 正则化损失系数
        regu_rate = 0.0001
        model = Model()
        x = tf.placeholder(tf.float32, [None, n_node_ary[0]], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, n_node_ary[-1]], name='y_input')
        if regu_bool:
            reguzation = tf.contrib.layers.l2_regularizer(regu_rate)
        else:
            reguzation = None
        y = model.get_model_result(reguzation, n_node_ary, x, act=act)
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope('moving_average'):
            avg_class = tf.train.ExponentialMovingAverage(0.99, global_step)
            avg_op = avg_class.apply(tf.trainable_variables())
        with tf.name_scope('loss_function'):
            loss = loss_fun(y_, y, False)
            if reguzation is None:
                total_loss = loss
            else:
                total_loss = loss + tf.add_n(tf.get_collection('losses'))
            tf.summary.scalar('loss', loss)
        with tf.name_scope('train_step'):
            learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                                       train_data_size / batch_size, learning_rate_decay)
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
            # train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
            with tf.control_dependencies([train_step, avg_op]):
                train_op = tf.no_op(name='train')
        merge = tf.summary.merge_all()

        with tf.Session() as sess:
            tf.local_variables_initializer().run()
            tf.initialize_all_variables().run()
            writer = tf.summary.FileWriter(model_log, sess.graph)
            # 生成结构图信息
            train_data, train_label = DataUtil.generator_train_data(train_file_ary, batch_size, feature_size)
            test_d, test_l = DataUtil.generator_test_data(test_file_ary, test_data_size, feature_size)
            coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(total_steps):
                e_batch, l_batch = sess.run([train_data, train_label])
                e_exam = [[0. if i.decode() == 'nan' else float(i.decode()) for i in x] for x in np.asarray(e_batch)]
                l_exam = [1 if x.decode('utf-8') == '1.0' else 0 for x in l_batch]
                label = tf.one_hot(l_exam, 2).eval()
                loss_value, step, _, summary = sess.run([total_loss, global_step, train_op, merge],
                                                        feed_dict={x: e_exam, y_: label})
                writer.add_summary(summary, i)
                if i % 500 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    loss_value, step, _, summary = sess.run([total_loss, global_step, train_op, merge],
                                                            feed_dict={x: e_exam, y_: label},
                                                            options=run_options, run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    a, b = sess.run([test_d, test_l])
                    test_data = [[0. if i.decode() == 'nan' else float(i.decode()) for i in x] for x in np.asarray(a)]
                    d = [1 if x.decode('utf-8') == '1.0' else 0 for x in b]
                    test_label = tf.one_hot(d, 2).eval()
                    accuracy_score = sess.run(accuracy, feed_dict={x: test_data, y_: test_label})
                    print('after training step %s 步，正确率 %f' % (i, accuracy_score))
            writer.close()
            model.save_model(sess, x, y, model_path)
            coord.request_stop()
            coord.join(threads)

