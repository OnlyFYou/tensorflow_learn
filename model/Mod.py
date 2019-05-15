import tensorflow as tf
from Base import Basic


class Model(Basic):
    # 获取权重和正则化损失
    def get_weights(self, shape, regulaizer):
        weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
        # 有正则化损失，加入losses集合
        if regulaizer is not None:
            tf.add_to_collection('losses', regulaizer(weights))
        return weights

    # 变量可视化
    def variables_summary(self, name, var):
        with tf.name_scope('summary'):
            tf.summary.histogram(name, var)
            # 平均值
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + name, mean)
            # 标准差
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev/' + name, stddev)

    # 神经网络层输出结果
    def nn_layer(self, input_tensor, input_dim, output_dim, avg_class, reguzation, layer_name, act):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.get_weights([input_dim, output_dim], reguzation)
                # self.variables_summary(layer_name + '/weights', weights)

            with tf.name_scope('biases'):
                # 偏置项不加入正则化损失
                biases = tf.Variable(tf.constant(0.1, shape=[output_dim]))
                # self.variables_summary(layer_name + '/biases', biases)

            with tf.name_scope('layer_result'):
                # 有滑动平均模型，计算滑动平均值
                if avg_class is None:
                    if act is None:
                        no_activation = tf.matmul(input_tensor, weights) + biases
                        # tf.summary.histogram('/no_activation', no_activation)
                        return no_activation
                    else:
                        activation = act(tf.matmul(input_tensor, weights) + biases, name='activation')
                        # tf.summary.histogram('/activation', activation)
                        return activation
                else:
                    if act is None:
                        no_activation = tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases)
                        # tf.summary.histogram(layer_name + '/avg/no_activation', no_activation)
                        return no_activation
                    else:
                        activation = act(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases), name='activation')
                        # tf.summary.histogram(layer_name + '/avg/activation', activation)
                        return activation
