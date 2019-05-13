import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Data:

    @staticmethod
    def origin_data(type, path):
        mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
        return mnist