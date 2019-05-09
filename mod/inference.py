import tensorflow as tf
import mod.util as util


# 神经网络层输出结果
def nn_layer(input_tensor, input_dim, output_dim, avg_class, reguzation, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = util.get_weights([input_dim, output_dim], reguzation)
            util.variables_summary(layer_name + '/weights', weights)

        with tf.name_scope('biases'):
            # 偏置项不加入正则化损失
            biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
            util.variables_summary(layer_name + '/biases', biases)

        with tf.name_scope('layer_result'):
            # 有滑动平均模型，计算滑动平均值
            if avg_class is None:
                no_activation = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram(layer_name + '/no_activation', no_activation)
                activation = act(no_activation, name='activation')
                tf.summary.histogram(layer_name + '/activation', activation)
            else:
                activation = tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases)
                tf.summary.histogram(layer_name + '/avg/no_activation', activation)
                activation = act(activation, name='activation')
                tf.summary.histogram(layer_name + '/avg/activation', activation)
        return activation
