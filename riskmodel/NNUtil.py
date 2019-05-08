import numpy as np
import tensorflow as tf
import scipy.stats as ss
from sklearn import metrics


class Losses:
    # 欧氏距离损失函数
    @staticmethod
    def mse(y, pred, _, weights = None):
        if weights is None:
            return tf.losses.mean_squared_error(y, pred)
        return tf.losses.mean_squared_error(y, pred, tf.reshape(weights, [-1, 1]))

    #交叉熵损失函数
    @staticmethod
    def cross_entropy(y, pred, already_prob, weights = None):
        if already_prob:
            eps = 1e-12
            # 如果已经是概率向量，就取对数， 从而后面softmax+crossentropy 能还原出pred本身
            pred = tf.log(tf.clip_by_value(pred, eps, 1-eps))
        if weights is None:
            return tf.losses.softmax_cross_entropy(y, pred)
        return tf.losses.softmax_cross_entropy(y, pred, weights)

    # 定义相关性系数损失函数
    @staticmethod
    def correlation(y, pred, weights=None):
        # 均值和方差
        y_mean, y_var = tf.nn.moments(y, 0)
        pred_mean, pred_var = tf.nn.moments(pred, 0)
        if weights is None:
            e = tf.reduce_mean((y-y_mean) * (pred-pred_mean))
        else:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean) * weights)
        # 将损失设置为负相关性， 期望模型的输出与标签的相关性增加
        return -e / tf.sqrt(y_var * pred_var)


# 定义能够根据模型预测和真实值来评估模型表现的类
class Metrics:
    """
        定义两个辅助字典
        sign_dict: key 为 metric名， value：正负1 1：说明metric越大越好
        require_prob: key 为 metric名， value为True 或 Fasle  True:该metric需要接受一个概率预测值 反之：接受一个类别预测值
    """
    sign_dict = {
        "f1_score":1, "r2_score":1, "auc":1, "multi_auc":1, "acc":1, "binary_acc":1,"mse":-1,"ber":-1, "los_coss":-1,
        "correction":1
    }
    require_prob = {name:False for name in sign_dict}
    require_prob["auc"] = True
    require_prob["multi_auc"] = True

    # 定义能够调整向量形状以适应metric函数输入要求的方法
    @staticmethod
    def check_shape(y, binary = False):
        y = np.asarray(y, np.float32)
        # 二位数组
        if len(y.shape) == 2:
            # 如果是二分类问题
            if binary:
                if y.shape[1] == 2:
                    return [..., 1]
                return y.ravel()
            # 如果不是而分类问题
            return np.argmax(y, axis=1)
        return y

    # 定义计算f1score的方法, 不均衡的而分类问题
    @staticmethod
    def f1_score(y, pred):
        return metrics.f1_score(Metrics.check_shape(y), Metrics.check_shape(pred))

    # 回归问题
    @staticmethod
    def r2_score(y, pred):
        return metrics.r2_score(y, pred)

    # 定义auc方法， 不均衡的二分类问题
    @staticmethod
    def auc(y, pred):
        return metrics.roc_auc_score(Metrics.check_shape(y, True), Metrics.check_shape(pred, True))

    # 定义计算多分类的auc方法，一般用于不均衡的多分类问题
    @staticmethod
    def mult_auc(y, pred):
        n_classes = pred.shape[1]
        if len(y.shape == 1):
            y = Toolbox.get_one_hot(y, n_classes)
        fpr, tpr = [None] * n_classes, [None] * n_classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ =metrics.roc_curve(y[:, i], pred[:, i])
        new_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        new_tpr = np.zeros_like(new_fpr)
        for i in range(n_classes):
            new_tpr += np.interp(new_fpr, fpr[i], tpr[i])
        new_tpr /= n_classes
        return metrics.auc(new_fpr, new_tpr)

    # 定义计算准确率的方法，一般用于均衡二分类问题与多分类问题
    @staticmethod
    def acc(y, pred):
        return np.mean(Metrics.check_shape(y) == Metrics.check_shape(pred))

    # 定义计算二分类准确率的方法
    @staticmethod
    def binary_acc(y, pred):
        return np.mean((Metrics.check_shape(y) > 0) == (Metrics.check_shape(pred) > 0))

    # 定义计算欧氏距离的方法
    @staticmethod
    def mse(y, pred):
        return np.mean(np.square(y.ravel() - pred.ravel()))

    # 定义计算log_loss的方法，一般用于分类
    @staticmethod
    def log_loss(y, pred):
        return metrics.log_loss(y, pred)

    # 定义计算相关性系数的方法，一般用于回归问题
    @staticmethod
    def correlation(y, pred):
        return float(ss.pearsonr(y, pred)[0])


# 定义能够输入x返回激活值的类
class Activations:
    @staticmethod
    def elu(x, name):
        return tf.nn.elu(x, name=name)

    @staticmethod
    def relu(x, name):
        return tf.nn.relu(x, name=name)

    @staticmethod
    def selu(x, name):
        return tf.nn.selu(x, name=name)

    @staticmethod
    def sigmoid(x, name):
        return tf.nn.sigmoid(x, name=name)

    @staticmethod
    def tanh(x, name):
        return tf.nn.tanh(x, name=name)

    @staticmethod
    def softplus(x, name):
        return tf.nn.softplus(x, name=name)

    @staticmethod
    def softmax(x, name):
        return tf.nn.softmax(x, name=name)

    @staticmethod
    def sign(x, name):
        return tf.sign(x, name=name)

    @staticmethod
    def ont_hot(x, name):
        return tf.multiply(
            x, tf.cast(tf.equal(x, tf.expand_dims(tf.reduce_max(x, 1), 1)), tf.float32),
            name=name
        )