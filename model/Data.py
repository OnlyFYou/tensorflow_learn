import tensorflow as tf
import numpy as np


class DataUtil:

    @staticmethod
    def get_data(path, feature_size, batch_size):
        filenames = path
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
        # 定义Reader
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        # 定义Decoder
        record_defaults = [['nan'] for _ in range(feature_size)]  # 解析为整数
        example_batch = tf.decode_csv(value, record_defaults=record_defaults)
        features = tf.stack(example_batch[0:-1])  # 前列数据，后1列标签
        label = tf.stack(example_batch[-1])
        example_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=batch_size, capacity= batch_size +50,
                                                            min_after_dequeue=batch_size+10, num_threads=20)
        return example_batch, label_batch

    @staticmethod
    def generator_data(path, sess, batch_size):
        example_batch, label_batch = DataUtil.get_data([path], 15,
                                                       batch_size)
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)
        e_batch, l_batch = sess.run([example_batch, label_batch])
        e_exam = [[0. if i.decode() == 'nan' else float(i.decode()) for i in x] for x in np.asarray(e_batch)]
        l_exam = [1 if x.decode('utf-8') == '1.0' else 0 for x in l_batch]
        label = tf.one_hot(l_exam, 2).eval()
        coord.request_stop()
        coord.join(threads)
        return e_exam, label
