import tensorflow as tf
'''
    模型持久化学习
'''
saver = tf.train.import_meta_graph("/path/to/model/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "/path/to/model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
