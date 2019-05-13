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

    def __init__(self, save_model_path):
        # 每步训练的数据量
        self._batch_size = None
        # 训练的迭代轮数
        self._n_iter = None
        # 测试数据构造器
        self._test_generator = None
        # 训练数据构造器
        self._train_generator = None
        self._generator_base = Generator
        # 输入特征维度
        self._n_dim = None
        # 输出维度
        self._n_class = None
        # 样本权重，用与解决不均衡数据，分类模型中，样本量少的权重大
        self._sample_weights = None
        self._save_model_path = save_model_path

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

    # 属性初始化 x, y 测试数据
    def init_from_data(self, x, y, x_test, y_test, sample_weights):
        self._sample_weights = sample_weights
        # 训练数据构造器
        self._train_generator = self._generator_base(x, y, "TrainGenerator", self._sample_weights, self._n_class)
        if x_test is not None and y_test is not None:
            # 测试数据构造器
            self._test_generator = self._generator_base(x_test, y_test, "TestGenerator", n_class=self._n_class)
        else:
            self._test_generator = None
        self._n_dim = self._train_generator.shape[-1]
        self._n_class = self._train_generator.n_class
        batch_size = 128
        self._batch_size = min(batch_size, len(self._train_generator))
        n_iter = -1
        if n_iter < 0:
            self._n_iter = int(len(self._train_generator) / batch_size)


# 构建数据预处理
class AutoBase:
    """
        name:模型的名字 data_info 数据超参数 pre_process_settings:数据预处理参数设置 nan_handler_settings：缺失值处理器参数设置
    """
    def __init__(self, name=None, data_info=None, pre_process_settings=None, nan_handler_settings=None):
        if name is None:
            raise ValueError("name should be provided when using AutoBase")
        self._name = name

        self._data_folder = None
        self.whether_redundant = None
        self.feature_sets = self.sparsity = self.class_prior = None
        self.n_features = self.all_num_idx = self.transform_dicts = None

        self.py_collections = []

        if data_info is None:
            data_info = {}
        else:
            assert_msg = "data_info should be a dictionary"
            assert isinstance(data_info, dict), assert_msg
        self.data_info = data_info
        self._data_info_initialized = False
        self.numerical_idx = self.categorical_columns = None

        if pre_process_settings is None:
            pre_process_settings = {}
        else:
            assert_msg = "pre_process_settings should be a dictionary"
            assert isinstance(pre_process_settings, dict), assert_msg
        self.pre_process_settings = pre_process_settings
        self._pre_processors = None
        self.pre_process_method = self.scale_method = self.reuse_mean_and_std = None

        if nan_handler_settings is None:
            nan_handler_settings = {}
        else:
            assert_msg = "nan_handler_settings should be a dictionary"
            assert isinstance(nan_handler_settings, dict), assert_msg
        self.nan_handler_settings = nan_handler_settings
        self._nan_handler = None
        self.nan_handler_method = self.reuse_nan_handler_values = None

        self.init_pre_process_settings()
        self.init_nan_handler_settings()

    @property
    def valid_numerical_idx(self):
        return np.array([
            is_numerical for is_numerical in self.numerical_idx
            if is_numerical is not None
        ])

    @property
    def valid_n_features(self):
        return np.array([
            n_feature for i, n_feature in enumerate(self.n_features)
            if self.numerical_idx[i] is not None
        ])

    @property
    def label2num_dict(self):
        return None if not self.transform_dicts[-1] else self.transform_dicts[-1]

    @property
    def num2label_dict(self):
        label2num_dict = self.label2num_dict
        if label2num_dict is None:
            return
        num_label_list = sorted([(i, c) for c, i in label2num_dict.items()])
        return np.array([label for _, label in num_label_list])

    # 初始化模型结构超参数
    def init_data_info(self):
        if self._data_info_initialized:
            return
        self._data_info_initialized = True
        self.numerical_idx = self.data_info.get("numerical_idx", None)
        self.categorical_columns = self.data_info.get("categorical_columns", None)
        self.feature_sets = self.data_info.get("feature_sets", None)
        self.sparsity = self.data_info.get("sparsity", None)
        self.class_prior = self.data_info.get("class_prior", None)
        if self.feature_sets is not None and self.numerical_idx is not None:
            self.n_features = [len(feature_set) for feature_set in self.feature_sets]
            self._gen_categorical_columns()
        self._data_folder = self.data_info.get("data_folder", "_Data")
        self.data_info.setdefault("file_type", "txt")
        self.data_info.setdefault("shuffle", True)
        self.data_info.setdefault("test_rate", 0.1)
        self.data_info.setdefault("stage", 3)

    # 数据准备
    def _auto_init_from_data(self, x, y, x_test, y_test, names):
        stage = self.data_info["stage"]
        shuffle = self.data_info["shuffle"]
        file_type = self.data_info["file_type"]
        test_rate = self.data_info["test_rate"]
        args = (self.numerical_idx, file_type, names, shuffle, test_rate, stage)
        if x is None or y is None:
            x, y, x_test, y_test = self._load_data(None, *args)
        else:
            data = np.hstack([x, y.reshape([-1, 1])])
            if x_test is not None and y_test is not None:
                data = (data, np.hstack([x_test, y_test.reshape([-1, 1])]))
            x, y, x_test, y_test = self._load_data(data, *args)
        self._handle_unbalance(y)
        self._handle_sparsity()
        return x, y, x_test, y_test

    def _load_data(self, data=None, numerical_idx=None, file_type="txt", names=("train", "test"),
                   shuffle=True, test_rate=0.1, stage=3):
        use_cached_data = False
        train_data = test_data = None
        data_cache_folder = os.path.join(self._data_folder, "_Cache", self._name)
        data_info_folder = os.path.join(self._data_folder, "_DataInfo")
        data_info_file = os.path.join(data_info_folder, "{}.info".format(self._name))
        train_data_file = os.path.join(data_cache_folder, "train.npy")
        test_data_file = os.path.join(data_cache_folder, "test.npy")

        if data is None and stage >= 2 and os.path.isfile(train_data_file):
            print("Restoring data")
            use_cached_data = True
            train_data = np.load(train_data_file)
            if not os.path.isfile(test_data_file):
                test_data = None
                data = train_data
            else:
                test_data = np.load(test_data_file)
                data = (train_data, test_data)
        if use_cached_data:
            n_train = None
        else:
            if data is None:
                is_ndarray = False
                data, test_rate = self._get_data_from_file(file_type, test_rate)
            else:
                is_ndarray = True
                if not isinstance(data, tuple):
                    test_rate = 0
                    data = np.asarray(data, dtype=np.float32)
                else:
                    data = tuple(
                        arr if isinstance(arr, list) else
                        np.asarray(arr, np.float32) for arr in data
                    )
            if isinstance(data, tuple):
                if shuffle:
                    np.random.shuffle(data[0]) if is_ndarray else random.shuffle(data[0])
                n_train = len(data[0])
                data = np.vstack(data) if is_ndarray else data[0] + data[1]
            else:
                if shuffle:
                    np.random.shuffle(data) if is_ndarray else random.shuffle(data)
                n_train = int(len(data) * (1 - test_rate)) if test_rate > 0 else -1

        if not os.path.isdir(data_info_folder):
            os.makedirs(data_info_folder)
        if not os.path.isfile(data_info_file) or stage == 1:
            print("Generating data info")
            if numerical_idx is not None:
                self.numerical_idx = numerical_idx
            elif self.numerical_idx is not None:
                numerical_idx = self.numerical_idx
            if not self.feature_sets or not self.n_features or not self.all_num_idx:
                is_regression = self.data_info.pop(
                    "is_regression",
                    numerical_idx is not None and numerical_idx[-1]
                )
                self.feature_sets, self.n_features, self.all_num_idx, self.numerical_idx = (
                    Toolbox.get_feature_info(data, numerical_idx, is_regression)
                )
            self.n_class = 1 if self.numerical_idx[-1] else self.n_features[-1]
            self._get_transform_dicts()
            with open(data_info_file, "wb") as file:
                pickle.dump([
                    self.n_features, self.numerical_idx, self.transform_dicts
                ], file)
        elif stage == 3:
            print("Restoring data info")
            with open(data_info_file, "rb") as file:
                info = pickle.load(file)
                self.n_features, self.numerical_idx, self.transform_dicts = info
            self.n_class = 1 if self.numerical_idx[-1] else self.n_features[-1]

        if not use_cached_data:
            if n_train > 0:
                train_data, test_data = data[:n_train], data[n_train:]
            else:
                train_data, test_data = data, None
            train_name, test_name = names
            train_data = self._transform_data(train_data, train_name, train_name, True, True, stage)
            if test_data is not None:
                test_data = self._transform_data(test_data, test_name, train_name, True, stage=stage)
        self._gen_categorical_columns()
        if not use_cached_data and stage == 3:
            print("Caching data...")
            if not os.path.isdir(data_cache_folder):
                os.makedirs(data_cache_folder)
            np.save(train_data_file, train_data)
            if test_data is not None:
                np.save(test_data_file, test_data)

        x, y = train_data[..., :-1], train_data[..., -1]
        if test_data is not None:
            x_test, y_test = test_data[..., :-1], test_data[..., -1]
        else:
            x_test = y_test = None
        self.sparsity = ((x == 0).sum() + np.isnan(x).sum()) / np.prod(x.shape)
        _, class_counts = np.unique(y, return_counts=True)
        self.class_prior = class_counts / class_counts.sum()

        self.data_info["numerical_idx"] = self.numerical_idx
        self.data_info["categorical_columns"] = self.categorical_columns

        return x, y, x_test, y_test

    def _get_data_from_file(self, file_type, test_rate, target=None):
        if file_type == "txt":
            sep, include_header = " ", False
        elif file_type == "csv":
            sep, include_header = ",", True
        else:
            raise NotImplementedError("File type '{}' not recognized".format(file_type))
        if target is None:
            target = os.path.join(self._data_folder, self._name)
        if not os.path.isdir(target):
            with open(target + ".{}".format(file_type), "r") as file:
                data = Toolbox.get_data(file, sep, include_header)
        else:
            with open(os.path.join(target, "train.{}".format(file_type)), "r") as file:
                train_data = Toolbox.get_data(file, sep, include_header)
            test_rate = 0
            test_file = os.path.join(target, "test.{}".format(file_type))
            if not os.path.isfile(test_file):
                data = train_data
            else:
                with open(test_file, "r") as file:
                    test_data = Toolbox.get_data(file, sep, include_header)
                data = (train_data, test_data)
        return data, test_rate

    # 去除冗余特征 该特征全不相同或全部相同 两种情况
    def _transform_data(self, data, name, train_name="train",
                        include_label=False, refresh_redundant_info=False, stage=3):
        print("Transforming {0}data{2} at stage {1}".format(
            "{} ".format(name), stage,
            "" if name == train_name or not self.reuse_mean_and_std else
            " with {} data".format(train_name),
        ))
        is_ndarray = isinstance(data, np.ndarray)
        if refresh_redundant_info or self.whether_redundant is None:
            self.whether_redundant = np.array([
                True if local_dict is None else False
                for local_dict in self.transform_dicts
            ])
        targets = [
            (i, local_dict) for i, (idx, local_dict) in enumerate(
                zip(self.numerical_idx, self.transform_dicts)
            ) if not idx and local_dict and not self.whether_redundant[i]
        ]
        if targets and targets[-1][0] == len(self.numerical_idx) - 1 and not include_label:
            targets = targets[:-1]
        if stage == 1 or stage == 3:
            # Transform data & Handle redundant
            n_redundant = np.sum(self.whether_redundant)
            if n_redundant == 0:
                whether_redundant = None
            else:
                whether_redundant = self.whether_redundant
                if not include_label:
                    whether_redundant = whether_redundant[:-1]
                if refresh_redundant_info:
                    warn_msg = "{} redundant: {}{}".format(
                        "These {} columns are".format(n_redundant) if n_redundant > 1 else "One column is",
                        [i for i, redundant in enumerate(whether_redundant) if redundant],
                        ", {} will be removed".format("it" if n_redundant == 1 else "they")
                    )
                    print(warn_msg)
            valid_indices = [
                i for i, redundant in enumerate(self.whether_redundant)
                if not redundant
            ]
            if not include_label:
                valid_indices = valid_indices[:-1]
            for i, line in enumerate(data):
                for j, local_dict in targets:
                    elem = line[j]
                    if isinstance(elem, str):
                        line[j] = local_dict.get(elem, local_dict.get("nan", len(local_dict)))
                    elif math.isnan(elem):
                        line[j] = local_dict["nan"]
                    else:
                        line[j] = local_dict.get(elem, local_dict.get("nan", len(local_dict)))
                if not is_ndarray and whether_redundant is not None:
                    data[i] = [line[j] for j in valid_indices]
            if is_ndarray and whether_redundant is not None:
                data = data[..., valid_indices].astype(np.float32)
            else:
                data = np.array(data, dtype=np.float32)
        if stage == 2 or stage == 3:
            data = np.asarray(data, dtype=np.float32)
            # Handle nan
            if self._nan_handler is None:
                self._nan_handler = NanHandler(
                    method=self.nan_handler_method,
                    reuse_values=self.reuse_nan_handler_values
                )
            data = self._nan_handler.transform(data, self.valid_numerical_idx[:-1])
            # Pre-process data
            if self._pre_processors is not None:
                pre_processor_name = train_name if self.reuse_mean_and_std else name
                pre_processor = self._pre_processors.setdefault(
                    pre_processor_name, PreProcessor(
                        self.pre_process_method, self.scale_method
                    )
                )
                if not include_label:
                    data = pre_processor.transform(data, self.valid_numerical_idx[:-1])
                else:
                    data[..., :-1] = pre_processor.transform(data[..., :-1], self.valid_numerical_idx[:-1])
        return data

    def _pop_preprocessor(self, name):
        if isinstance(self._pre_processors, dict) and name in self._pre_processors:
            self._pre_processors.pop(name)

    def get_transformed_data_from_file(self, file, file_type="txt", include_label=False):
        x, _ = self._get_data_from_file(file_type, 0, file)
        x = self._transform_data(x, "new", include_label=include_label)
        self._pop_preprocessor("new")
        return x

    def get_labels_from_classes(self, classes):
        num2label_dict = self.num2label_dict
        if num2label_dict is None:
            return classes
        return num2label_dict[classes]

    def predict_labels(self, x):
        return self.get_labels_from_classes(self.predict_classes(x))

    def predict_classes(self, x):
        raise ValueError

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

    def init_pre_process_settings(self):
        self.pre_process_method = self.pre_process_settings.setdefault("pre_process_method", "normalize")
        self.scale_method = self.pre_process_settings.setdefault("scale_method", "truncate")
        self.reuse_mean_and_std = self.pre_process_settings.setdefault("reuse_mean_and_std", False)
        if self.pre_process_method is not None and self._pre_processors is None:
            self._pre_processors = {}

    def init_nan_handler_settings(self):
        self.nan_handler_method = self.nan_handler_settings.setdefault("nan_handler_method", "median")
        self.reuse_nan_handler_values = self.nan_handler_settings.setdefault("reuse_nan_handler_values", True)

    def _gen_categorical_columns(self):
        self.categorical_columns = [
            (i, value) for i, value in enumerate(self.valid_n_features)
            if not self.valid_numerical_idx[i] and self.valid_numerical_idx[i] is not None
        ]
        if not self.valid_numerical_idx[-1]:
            self.categorical_columns.pop()

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


class Generator:
    # x, y 测试数据占位符
    def __init__(self, x, y, name="Generator", weights=None, n_class=None, shuffle=True):
        self._cache = {}
        self._x, self._y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if weights is None:
            self._sample_weights = None
        else:
            self._sample_weights = np.asarray(weights, np.float32)
        if n_class is not None:
            self.n_class = n_class
        else:
            y_int = self._y.astype(np.int32)
            if np.allclose(self._y, y_int):
                # 分类问题
                assert y_int.min() == 0, "Labels should start from 0"
                self.n_class = y_int.max() + 1
            else:
                # 回归问题
                self.n_class = 1
        self._name = name
        self._do_shuffle = shuffle
        self._all_valid_data = self._generate_all_valid_data()
        self._valid_indices = np.arange(len(self._all_valid_data))
        self._random_indices = self._valid_indices.copy()
        np.random.shuffle(self._random_indices)
        self._batch_cursor = -1

    def __enter__(self):
        self._cache_current_status()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._restore_cache()

    def __getitem__(self, item):
        return getattr(self, "_" + item)

    def __len__(self):
        return self.n_valid

    def __str__(self):
        return "{}_{}".format(self._name, self.shape)

    __repr__ = __str__

    @property
    def n_valid(self):
        return len(self._valid_indices)

    @property
    def n_dim(self):
        return self._x.shape[-1]

    @property
    def shape(self):
        return self.n_valid, self.n_dim

    def _generate_all_valid_data(self):
        # _x _y(变为一列N行) 水平平铺
        return np.hstack([self._x, self._y.reshape([-1, 1])])

    def _cache_current_status(self):
        self._cache["_valid_indices"] = self._valid_indices
        self._cache["_random_indices"] = self._random_indices

    def _restore_cache(self):
        self._valid_indices = self._cache["_valid_indices"]
        self._random_indices = self._cache["_random_indices"]
        self._cache = {}

    def set_indices(self, indices):
        indices = np.asarray(indices, np.int)
        self._valid_indices = self._valid_indices[indices]
        self._random_indices = self._random_indices[indices]

    def set_range(self, start, end=None):
        if end is None:
            self._valid_indices = self._valid_indices[start:]
            self._random_indices = self._random_indices[start:]
        else:
            self._valid_indices = self._valid_indices[start:end]
            self._random_indices = self._random_indices[start:end]

    def get_indices(self, indices):
        return self._get_data(np.asarray(indices, np.int))

    def get_range(self, start, end=None):
        if end is None:
            return self._get_data(self._valid_indices[start:])
        return self._get_data(self._valid_indices[start:end])

    def _get_data(self, indices, return_weights=True):
        data = self._all_valid_data[indices]
        if not return_weights:
            return data
        weights = None if self._sample_weights is None else self._sample_weights[indices]
        return data, weights

    def gen_batch(self, n_batch, re_shuffle=True):
        n_batch = min(n_batch, self.n_valid)
        logger = logging.getLogger("DataReader")
        if n_batch == -1:
            n_batch = self.n_valid
        if self._batch_cursor < 0:
            self._batch_cursor = 0
        if self._do_shuffle:
            if self._batch_cursor == 0 and re_shuffle:
                logger.debug("Re-shuffling random indices")
                np.random.shuffle(self._random_indices)
            indices = self._random_indices
        else:
            indices = self._valid_indices
        logger.debug("Generating batch with size={}".format(n_batch))
        end = False
        next_cursor = self._batch_cursor + n_batch
        if next_cursor >= self.n_valid:
            next_cursor = self.n_valid
            end = True
        data, w = self._get_data(indices[self._batch_cursor:next_cursor])
        self._batch_cursor = -1 if end else next_cursor
        logger.debug("Done")
        return data, w

    def gen_random_subset(self, n):
        n = min(n, self.n_valid)
        logger = logging.getLogger("DataReader")
        logger.debug("Generating random subset with size={}".format(n))
        start = random.randint(0, self.n_valid - n)
        subset, weights = self._get_data(self._random_indices[start:start + n])
        logger.debug("Done")
        return subset, weights

    def get_all_data(self, return_weights=True):
        if self._all_valid_data is not None:
            if return_weights:
                return self._all_valid_data, self._sample_weights
            return self._all_valid_data
        return self._get_data(self._valid_indices, return_weights)