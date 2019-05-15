import tensorflow as tf
import numpy as np


class DataUtil:

    @staticmethod
    def get_data(path, feature_size, batch_size):
        filenames = path
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False,)
        # 定义Reader
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        # 定义Decoder
        record_defaults = [['nan'] for _ in range(feature_size)]  # 解析为整数
        example_batch = tf.decode_csv(value, record_defaults=record_defaults)
        features = tf.stack(example_batch[0:-1])  # 前列数据，后1列标签
        label = tf.stack(example_batch[-1])
        example_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=batch_size, capacity=3000,
                                                            min_after_dequeue=1000, num_threads=12)
        return example_batch, label_batch

    @staticmethod
    def generator_train_data(train_path, train_batch_size, feature_size):
        example_batch, label_batch = DataUtil.get_data(train_path, feature_size,
                                                       train_batch_size)

        return example_batch, label_batch

    @staticmethod
    def generator_test_data(test_path, test_data_size, feature_size):
        test_data, test_label = DataUtil.get_data(test_path, feature_size,
                                                  test_data_size)
        return test_data, test_label
