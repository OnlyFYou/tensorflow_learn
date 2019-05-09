import tensorflow as tf


# 获取权重和正则化损失
def get_weights(shape, regulaizer):
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))
    # 有正则化损失，加入losses集合
    if regulaizer is not None:
        tf.add_to_collection('losses', regulaizer(weights))
    return weights


# 变量可视化
def variables_summary(name, var):
    with tf.name_scope('summary'):
        tf.summary.histogram(name, var)
        # 平均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

