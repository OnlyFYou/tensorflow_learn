import tensorflow as tf
import logging

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def


# 模型基础抽象
class Base:

    def __init__(self, save_model_path):
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