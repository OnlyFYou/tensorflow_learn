import tensorflow as tf
from mod import Base
import math
import unicodedata


# 损失函数
class Losses(Base):

    # 欧式距离
    @staticmethod
    def mse(y, pred, _, weights=None):
        if weights is None:
            return tf.losses.mean_squared_error(y, pred)
        return tf.losses.mean_squared_error(y, pred, tf.reshape(weights, [-1, 1]))

    # 交叉熵 pred：输出结果  y:正确结果
    @staticmethod
    def cross_entropy(y, pred, already_prob, argmax=False, weights=None):
        # 神经网络输出的已经是经过softMax()的概率
        if already_prob:
            eps = 1e-12
            pred = tf.log(tf.clip_by_value(pred, eps, 1 - eps))
            return -tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-12, 1.0)))
        if weights is None:
            if argmax:
                return tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=tf.argmax(y, axis=1))
            else:
                return tf.losses.softmax_cross_entropy(labels=y, logits=pred)
        if argmax:
            return tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=tf.argmax(y, axis=1), weights=weights)
        else:
            return tf.losses.softmax_cross_entropy(labels=y, logits=pred, weights=weights)

    # 相关性系数损失函数
    @staticmethod
    def correlation(y, pred, _, weights=None):
        # 均值和方差
        y_mean, y_var = tf.nn.moments(y, 0)
        pred_mean, pred_var = tf.nn.moments(pred, 0)
        if weights is None:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean))
        else:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean) * weights)
        return -e / tf.sqrt(y_var * pred_var)


# 激活函数
class Activations(Base):

    @staticmethod
    def elu(x, name):
        return tf.nn.elu(x, name)

    @staticmethod
    def relu(x, name):
        return tf.nn.relu(x, name)

    @staticmethod
    def selu(x, name):
        return tf.nn.sele(x, name)

    @staticmethod
    def sigmoid(x, name):
        return tf.nn.sigmoid(x, name)

    @staticmethod
    def tanh(x, name):
        return tf.nn.tanh(x, name)

    @staticmethod
    def softplus(x, name):
        return tf.nn.softplus(x, name)

    @staticmethod
    def softmax(x, name):
        return tf.nn.softmax(x, name=name)

    @staticmethod
    def sign(x, name):
        return tf.sign(x, name)

    @staticmethod
    def one_hot(x, name):
        return tf.multiply(
            x,
            tf.cast(tf.equal(x, tf.expand_dims(tf.reduce_max(x, 1), 1)), tf.float32),
            name=name
        )


class Toolbox:
    @staticmethod
    def is_number(s):
        try:
            s = float(s)
            if math.isnan(s):
                return False
            return True
        except ValueError:
            try:
                unicodedata.numeric(s)
                return True
            except (TypeError, ValueError):
                return False

    @staticmethod
    def all_same(target):
        x = target[0]
        for new in target[1:]:
            if new != x:
                return False
        return True

    @staticmethod
    def all_unique(target):
        seen = set()
        return not any(x in seen or seen.add(x) for x in target)