import numpy as np
import tensorflow as tf
import math
import unicodedata
# 损失函数


class Losses:

    # 欧式距离
    @staticmethod
    def mse(y, pred, weights=None):
        if weights is None:
            return tf.losses.mean_squared_error(y, pred)
        return tf.losses.mean_squared_error(y, pred, tf.reshape(weights, [-1, 1]))

    # 交叉熵 pred：输出结果  y:正确结果
    @staticmethod
    def cross_entropy(label, pred, already_softmax, argmax=False, weights=None):
        # 神经网络输出的已经是经过softMax()的概率
        if already_softmax:
            return -tf.reduce_mean(label * tf.log(tf.clip_by_value(pred, 1e-12, 1.0)))
        if weights is None:
            if argmax:
                return tf.losses.sparse_softmax_cross_entropy(logits=pred, onehot_labels=tf.argmax(label, axis=1))
            else:
                return tf.losses.softmax_cross_entropy(logits=pred, onehot_labels=label)
        if argmax:
            return tf.losses.sparse_softmax_cross_entropy(logits=pred, onehot_labels=tf.argmax(label, axis=1), weights=weights)
        else:
            return tf.losses.softmax_cross_entropy(onehot_labels=label, logits=pred, weights=weights)

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
class Activations:

    @staticmethod
    def elu():
        return tf.nn.elu

    @staticmethod
    def relu(x, name):
        return tf.nn.relu(x, name)

    @staticmethod
    def selu():
        return tf.nn.selu

    @staticmethod
    def sigmoid():
        return tf.nn.sigmoid

    @staticmethod
    def tanh():
        return tf.nn.tanh

    @staticmethod
    def softplus():
        return tf.nn.softplus

    @staticmethod
    def softmax():
        return tf.nn.softmax

    @staticmethod
    def sign():
        return tf.sign

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

    # 从文件中获取数据
    @staticmethod
    def get_data(file, sep=" ", include_header=False, logger=None):
        msg = "Fetching data"
        print(msg) if logger is None else logger.debug(msg)
        data = [[elem if elem else "nan" for elem in line.strip().split(sep)] for line in file]
        if include_header:
            return data[1:]
        return data

    @staticmethod
    def warn_all_same(i, logger=None):
        warn_msg = "All values in column {} are the same, it'll be treated as redundant".format(i)
        print(warn_msg) if logger is None else logger.debug(warn_msg)

    @staticmethod
    def warn_all_unique(i, logger=None):
        warn_msg = "All values in column {} are unique, it'll be treated as redundant".format(i)
        print(warn_msg) if logger is None else logger.debug(warn_msg)

    @staticmethod
    def pop_nan(feat):
        no_nan_feat = []
        for f in feat:
            try:
                f = float(f)
                if math.isnan(f):
                    continue
                no_nan_feat.append(f)
            except ValueError:
                no_nan_feat.append(f)
        return no_nan_feat

    @staticmethod
    def shrink_nan(feat):
        new = np.asarray(feat, np.float32)
        new = new[~np.isnan(new)].tolist()
        if len(new) < len(feat):
            new.append(float("nan"))
        return new


# 缺失值处理
class NanHandler:
    def __init__(self, method, reuse_values=True):
        self._values = None
        self.method = method
        self.reuse_values = reuse_values

    def transform(self, x, numerical_idx, refresh_values=False):
        if self.method is None:
            pass
        elif self.method == "delete":
            x = x[~np.any(np.isnan(x[..., numerical_idx]), axis=1)]
        else:
            if self._values is None:
                self._values = [None] * len(numerical_idx)
            for i, (v, numerical) in enumerate(zip(self._values, numerical_idx)):
                if not numerical:
                    continue
                feat = x[..., i]
                mask = np.isnan(feat)
                if not np.any(mask):
                    continue
                if self.reuse_values and not refresh_values and v is not None:
                    new_value = v
                else:
                    new_value = getattr(np, self.method)(feat[~mask])
                    if self.reuse_values and (v is None or refresh_values):
                        self._values[i] = new_value
                feat[mask] = new_value
        return x

    def reset(self):
        self._values = None


# 连续型特征的数据预处理 标准化
class PreProcessor:
    def __init__(self, method, scale_method, eps_floor=1e-4, eps_ceiling=1e12):
        self.method, self.scale_method = method, scale_method
        self.eps_floor, self.eps_ceiling = eps_floor, eps_ceiling
        self.redundant_idx = None
        self.min = self.max = self.mean = self.std = None

    def _scale(self, x, numerical_idx):
        targets = x[..., numerical_idx]
        self.redundant_idx = [False] * len(numerical_idx)
        mean = std = None
        if self.mean is not None:
            mean = self.mean
        if self.std is not None:
            std = self.std
        if self.min is None:
            self.min = targets.min(axis=0)
        if self.max is None:
            self.max = targets.max(axis=0)
        if mean is None:
            mean = targets.mean(axis=0)
        abs_targets = np.abs(targets)
        max_features = abs_targets.max(axis=0)
        if self.scale_method is not None:
            max_features_res = max_features - mean
            mask = max_features_res > self.eps_ceiling
            n_large = np.sum(mask)
            if n_large > 0:
                idx_lst, val_lst = [], []
                mask_cursor = -1
                for i, numerical in enumerate(numerical_idx):
                    if not numerical:
                        continue
                    mask_cursor += 1
                    if not mask[mask_cursor]:
                        continue
                    idx_lst.append(i)
                    val_lst.append(max_features_res[mask_cursor])
                    local_target = targets[..., mask_cursor]
                    local_abs_target = abs_targets[..., mask_cursor]
                    sign_mask = np.ones(len(targets))
                    sign_mask[local_target < 0] *= -1
                    scaled_value = self._scale_abs_features(local_abs_target) * sign_mask
                    targets[..., mask_cursor] = scaled_value
                    if self.mean is None:
                        mean[mask_cursor] = np.mean(scaled_value)
                    max_features[mask_cursor] = np.max(scaled_value)
                warn_msg = "{} value which is too large: [{}]{}".format(
                    "These {} columns contain".format(n_large) if n_large > 1 else "One column contains",
                    ", ".join(
                        "{}: {:8.6f}".format(idx, val)
                        for idx, val in zip(idx_lst, val_lst)
                    ),
                    ", {} will be scaled by '{}' method".format(
                        "it" if n_large == 1 else "they", self.scale_method
                    )
                )
                print(warn_msg)
                x[..., numerical_idx] = targets
        if std is None:
            if np.any(max_features > self.eps_ceiling):
                targets = targets - mean
            std = np.maximum(self.eps_floor, targets.std(axis=0))
        if self.mean is None and self.std is None:
            self.mean, self.std = mean, std
        return x

    def _scale_abs_features(self, abs_features):
        if self.scale_method == "truncate":
            return np.minimum(abs_features, self.eps_ceiling)
        if self.scale_method == "divide":
            return abs_features / self.max
        if self.scale_method == "log":
            return np.log(abs_features + 1)
        return getattr(np, self.scale_method)(abs_features)

    def _normalize(self, x, numerical_idx):
        x[..., numerical_idx] -= self.mean
        x[..., numerical_idx] /= np.maximum(self.eps_floor, self.std)
        return x

    def _min_max(self, x, numerical_idx):
        x[..., numerical_idx] -= self.min
        x[..., numerical_idx] /= np.maximum(self.eps_floor, self.max - self.min)
        return x

    def transform(self, x, numerical_idx):
        x = self._scale(np.array(x, dtype=np.float32), numerical_idx)
        x = getattr(self, "_" + self.method)(x, numerical_idx)
        return x
