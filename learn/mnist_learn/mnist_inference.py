# _*_ coding: utf-8 _*_
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 获取神经网络参数，把参数加入正则化损失
def get_weights_variable(shape, regulaizer):
    weights = tf.get_variable('weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regulaizer is not None:
        tf.add_to_collection('losses', regulaizer(weights))
    return weights


# 前向传播结果
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weights_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weights_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2
