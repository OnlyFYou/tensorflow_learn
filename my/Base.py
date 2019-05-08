import tensorflow as tf
from tensorflow.contrib.eager.python.examples.gan.mnist import Generator


class Base:
    # 定义模型的签名
    signature = 'Base'
    '''
        初始化结构
        name:模型名字
        model_param_settings: 管理模型超参数字典
        model_structure_settings: 管理结构超参数字典
    '''
    def __init__(self, name=None, model_param_settings=None, model_structure_settings=None):
        # 定义一个字典存储训练过程中的中间结果
        self.log = {}
        self._name = name
        # 定义名字后缀,用于区分不同的模型
        self._name_appendix = ''
        # 定义是否已经初始化超参数属性
        self._setting_initialized = False

        # 定义数据生成器种类
        self._generator_base = Generator
        # 分别定义代指训练，交叉验证所用到的数据生成器的属性
        self._train_generator = self._test_generator = None
        # 分别定义代指样本权重以及计算图中的样本权重对应的placeholder属性
        self._same_weights = self._tf_sample_weights = None
        # 分别定义记录特征维度与类别数目的属性 当问题是回归问题时，self.n_class = 1
        self.n_dim = self.n_class = None
        self.n_random_train_subset = self.n_random_test_subset = None

        # 处理模型超参数
        if model_param_settings is None:
            self.model_param_settings = {}
        else:
            assert_msg = 'model_param_settings should be a dictionary'
            assert isinstance(model_param_settings, dict), assert_msg
            self.model_param_settings = model_param_settings
        self.lr = None
        self._loss = self._loss_name = self._metric_name = None
        self._optimizer_name = self._optimizer = None
        self.n_epoch = self.max_epoch = self.n_iter = self.batch_size = None

        # 处理结构超参数
        if model_structure_settings is None:
            self.model_param_settings = {}
        else:
            assert_msg = 'model_structure_settings should be a dictionary'
            assert isinstance(model_structure_settings, dict), assert_msg
            self.model_structure_settings = model_structure_settings

        # 定义一些辅助模型保存，复用的属性
        self._model_built = False
        self.py_collections = self.tf_collections = None
        self._define_py_collections()
        self._define_tf_collections()

        # 定义模型参数相关的属性 self._ws 权值矩阵  self._bs 偏置矩阵
        self._ws, self._bs = [], []
        # self._is_training 标识当前是否是训练过程
        self._is_training = tf.placeholder(tf.bool, name='is_training')
        self._loss = self._train_step = None
        self._tfx = self._tfy = self._output = self._prob_output = None

        self._sess = None
        self._graph = tf.Graph()
        self._sess_config = self.model_param_settings.pop('sess_config', None)

