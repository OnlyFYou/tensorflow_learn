import tensorflow as tf
import numpy as np
from Data import DataUtil
with tf.Session() as sess:

    train_e, train_l = DataUtil.generator_data(sess, 52)