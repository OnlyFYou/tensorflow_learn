import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = '/path/to/mylog'
BATCH_SIZE = 100
TRAIN_STEP = 3000


# 生成变量监控信息，并定义生成监控信息日志的操作
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        # 变量分布的直方图信息
        tf.summary.histogram(name, var)
        # 平均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


# 生成一个全链接神经网络 input_dim:输入维度  output_dim:输出维度 act:激活函数
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 每一层放入一个命名空间
    with tf.name_scope(layer_name):
        # 权重
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')

        # 偏置项
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        # 输出结果
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)
        # 改层输出结果
        return activations


def main(_):
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)

    # 定义输入
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

        # 姜输入向量还原成图片的像素矩阵，写入日志
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, 10)

        hidden1 = nn_layer(x, 784, 500, 'layer1')
        y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
            tf.summary.scalar('cross entropy', cross_entropy)
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
            tf.global_variables_initializer().run()

            for i in range(TRAIN_STEP):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)

                summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys})
                summary_writer.add_run_metadata(summary, i)
        summary_writer.close()


if __name__ == '__main__':
    tf.app.run()
