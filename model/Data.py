import tensorflow as tf


class DataUtil:

    @staticmethod
    def get_data(path, feature_size, batch_size):
        filenames = path
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
        # 定义Reader
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)
        # 定义Decoder
        record_defaults = [['non'] for _ in range(feature_size)]  # 解析为整数
        example_batch = tf.decode_csv(value, record_defaults=record_defaults)
        features = tf.stack(example_batch[0:-1])  # 前列数据，后1列标签
        label = tf.stack(example_batch[-1:-2])
        example_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=batch_size, capacity=200,
                                                            min_after_dequeue=100, num_threads=6)
        return example_batch, label_batch

    if __name__ == "__main__":
        example_batch, label_batch = get_data( ['D:\\tensorflow_learn\\model\data\\A.csv'], 15, 5)
        # 运行Graph
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
            threads = tf.train.start_queue_runners(coord=coord)
            e_val, l_val = sess.run([example_batch, label_batch])
            label = tf.one_hot(tf.constant(l_val.astype(int)), 2)
            print(label)
            coord.request_stop()
            coord.join(threads)