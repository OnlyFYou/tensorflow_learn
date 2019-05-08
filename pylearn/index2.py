import tensorflow as tf

with tf.Session() as sess:
    print(sess.run(tf.log(10.0)))
