import numpy as np
import tensorflow as tf
import scipy.stats as ss
from sklearn import metrics
import my.Toolbox as tb


class Losses:
    # 定义欧式距离损失函数
    @staticmethod
    def mse(y, pred, _, weights=None):
        if weights is None:
            return tf.losses.mean_squared_error(y, pred)
        return tf.losses.mean_squared_error(y, pred, tf.reshape(weights, [-1, 1]))

    # 定义交叉熵损失函数
    # 需要给句 pred 是否已经是概率向量来调整损失函数的计算方法
    @staticmethod
    def corss_entropy(y, pred, already_prob, weights=None):
        if already_prob:
            eps = 1e-12
            pred = tf.log(tf.clip_by_value(pred, eps, 1 - eps))
        if weights is None:
            return tf.losses.softmax_cross_entropy(y, pred)
        return tf.losses.softmax_cross_entropy(y, pred, weights)

    # 定义相关性系数损失函数
    @staticmethod
    def correlation(y, pred, _, weights=None):
        # 利用tf.nn.moments 算出均值与方差
        y_mean, y_var = tf.nn.moments(y, 0)
        pred_mean, pred_var = tf.nn.moments(pred, 0)
        # 利用君主方差和相应公式算出相关性系数
        if weights is None:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean))
        else:
            e = tf.reduce_mean((y - y_mean) * (pred - pred_mean) * weights)
        # 将损失函数设置为负相关性，从而期望模型输出与标签的相关性增加
        return -e / tf.square(y_var * pred_var)


# 定义能够根据模型预测与真值来评估模型表现的类
class Metrics:
    """
        定义两个辅助字典
        sign_dict: key是metric名字， value为正负1，其中 1说明该metric越大越好 反之相反
        require_prob: key是metric名字，value为true 或 false 其中true说明该metric需要接受一个概率性预测，反之类别预测
    """
    sign_dict = {
        "f1_score": 1,
        "f2_score": 1,
        "auc": 1, "multi_auc": 1, "acc": 1, "binary_acc": 1,
        "mse": -1, "ber": -1, "log_loss": -1, "correlation": 1
    }
    require_prob = {
        name: False for name in sign_dict
    }
    require_prob["auc"] = True
    require_prob["multi_auc"] = True

    # 定义能够调整向量形状以适应metric函数的输入要求的方法， 由于scikit-learn 的 metric 中定义的函数接收的参数都是一维数组
    # 所以该方法的主要目的是把二维数组转为合乎要求的相应的一维数组
    @staticmethod
    def check_shape(y, binary=False):
        y = np.asarray(y, np.float32)
        # 当y是二维数组
        if len(y.shape) == 2:
            # 如果是二分类问题
            if binary:
                # 如果还是二维数组
                if y.shape[1] == 2:
                    # 返回第二列响应预测值
                    return y[..., 1]
                return y.ravel()
            # 如果不是二分类问题
            return y.argmax(y, axis=1)
        # 当不是二维数组
        return y

    # 定义计算f1scroe的方法，一般用于不均衡的二分类问题
    @staticmethod
    def f1_score(y, pred):
        return metrics.f1_score(Metrics.check_shape(y), Metrics.check_shape(pred))

    # 定义计算r2_score 的方法一般用户回归问题
    @staticmethod
    def r2_score(y, pred):
        return metrics.r2_score(y, pred)

    # 定义计算auc的方法 一般用于不均衡的二分类问题
    @staticmethod
    def auc(y, pred):
        return metrics.roc_auc_score(
            Metrics.check_shape(y, True),
            Metrics.check_shape(pred, True)
        )

    # 定义计算多分类auc的方法， 一般用户不均衡的多分类问题
    @staticmethod
    def multi_auc(y, pred):
        n_classes = pred.shape[1]
        if len(y.shape) == 1:
            y = tb.get_one_hot(y, n_classes)
        fpr, tpr = [None] * n_classes, [None] * n_classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve((y[:, 1], pred[:, 1]))
        new_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        new_tpr = np.zeros_like(new_fpr)
        for i in range(n_classes):
            new_tpr += np.interp(new_fpr, fpr[i], tpr[i])
        new_tpr /= n_classes
        return metrics.auc(new_fpr, new_tpr)

    # 定义计算准确率的方法，一般用于均衡二分类问题和多分类问题
    @staticmethod
    def acc(y, pred):
        return np.mean(Metrics.check_shape(y) == Metrics.check_shape(pred))

    # 定义计算二分类准确率的方法
    @staticmethod
    def binary_acc(y, pred):
        return np.mean(
            (Metrics.check_shape(y) > 0 == (Metrics.check_shape(pred) > 0))
        )

    # 定义计算平均距离的方法，一般用于回归问题
    @staticmethod
    def mse(y, pred):
        return np.mean(np.square(y.ravel() - pred.ravel()))

    # 定义计算log_loss的方法一般用于分类问题
    @staticmethod
    def log_loss(y, pred):
        return metrics.log_loss(y, pred)

    # 定义计算线性相关系数的方法，一般用于回归问题
    @staticmethod
    def correlation(y, pred):
        return float(ss.pearsonr(y, pred)[0])


# 定义能够根据输入x返回激活值的类
class Activations:
    @staticmethod
    def elu(x, name):
        return tf.nn.elu(x, name)

    @staticmethod
    def relu(x, name):
        return tf.nn.relu(x, name)

    @staticmethod
    def selu(x, name):
        return tf.nn.selu(x, name)

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
        return tf.nn.softplus(x, name)

    @staticmethod
    def sign(x, name):
        return tf.sign(x, name)

    # 定义ont_hot激活函数
    @staticmethod
    def ont_hot(x, name):
        return tf.multiply(
            x, tf.cast(tf.expand_dims(tf.reduce_max(x, 1), 1), tf.float32), name=name
        )