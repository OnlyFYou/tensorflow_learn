import os
import pickle
import random

import math
import numpy as np
import tensorflow as tf
import logging

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from Util import Toolbox, NanHandler, PreProcessor


class Basic:

    def save_model(self, sess, x, y):
        builder = tf.saved_model.builder.SavedModelBuilder(self._save_model_path)
        signature = predict_signature_def(inputs={'myInput': x},
                                          outputs={'myOutput': y})
        builder.add_meta_graph_and_variables(sess=sess,
                                             tags=[tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()

    def load_model(self, sess):
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], self._save_model_path)
        x = sess.graph.get_tensor_by_name('myInput:0')
        y = sess.graph.get_tensor_by_name('myOutput:0')
        return x, y

    @staticmethod
    def log_info(info):
        logging.info(info)

    @staticmethod
    def log_error(error):
        logging.error(error)



    # 不均衡数据处理
    def _handle_unbalance(self, y):
        if self.n_class == 1:
            return
        class_ratio = self.class_prior.min() / self.class_prior.max()
        if class_ratio < 0.1:
            warn_msg = "Sample weights will be used since class_ratio < 0.1 ({:8.6f})".format(class_ratio)
            print(warn_msg)
            if self._sample_weights is None:
                print("Sample weights are not provided, they'll be generated automatically")
                self._sample_weights = np.ones(len(y)) / self.class_prior[y.astype(np.int)]
                self._sample_weights /= self._sample_weights.sum()
                self._sample_weights *= len(y)

    # 稀疏数据的处理 数据特征含有非常多的0或者有非常多的缺失值  Dropout
    def _handle_sparsity(self):
        if self.sparsity >= 0.75:
            warn_msg = "Dropout will be disabled since data sparsity >= 0.75 ({:8.6f})".format(self.sparsity)
            print(warn_msg)
            self.dropout_keep_prob = 1.

    # 非数值型特征数值化 如果已经是连续性特征，该位置为空字典
    def _get_transform_dicts(self):
        # 非数值型特征数值化列表
        self.transform_dicts = [
            None if is_numerical is None else
            {key: i for i, key in enumerate(sorted(feature_set))}
            if not is_numerical and (not all_num or not np.allclose(
                np.sort(np.array(list(feature_set), np.float32).astype(np.int32)),
                np.arange(0, len(feature_set))
            )) else {} for is_numerical, feature_set, all_num in zip(
                self.numerical_idx[:-1], self.feature_sets[:-1], self.all_num_idx[:-1]
            )
        ]
        # 如果是回归问题，标签就是连续型标签， 直接加入一个空字典
        if self.n_class == 1:
            self.transform_dicts.append({})
        else:
            self.transform_dicts.append(self._get_label_dict())

    # 用于转换标签的转换字典
    def _get_label_dict(self):
        # 获取标签的所有取值
        labels = self.feature_sets[-1]
        sorted_labels = sorted(labels)
        # 如果标签并非全是数值型标签，就直接返回相应的转换字典
        if not all(Toolbox.is_number(str(label)) for label in labels):
            return {key: i for i, key in enumerate(sorted_labels)}
        if not sorted_labels:
            return {}
        numerical_labels = np.array(sorted_labels, np.float32)
        if numerical_labels.max() - numerical_labels.min() != self.n_class - 1:
            return {key: i for i, key in enumerate(sorted_labels)}
        return {}
