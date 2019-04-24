import tensorflow as tf
from numpy.random import RandomState
'''
    全连接神经网络
'''
w1 = tf.Variable(tf.random_normal([2, 4], stddev = 2, seed = 1))
w2 = tf.Variable(tf.random_normal([4, 1], stddev = 3, seed = 1))
x = tf.placeholder(tf.float32, shape=(None, 2), name = 'x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name = 'y-input')

'''
神经网络前向传播过程
'''
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

'''
损失函数
'''
y = tf.sigmoid(y)

'''
反向传播 交叉熵
'''
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-1, 1.0)) + (1-y_) * tf.log(tf.clip_by_value(1-y, 1e-10,1.0)))
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128

'''
随机生成训练数据
'''
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 <1)] for (x1, x2) in X]

'''
训练过程
'''
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(w1)
    print(w2)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * 8) % dataset_size
        end = min(start+8, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_:Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x: X, y_:Y})
            print(total_cross_entropy)
            print("\n")
    print(sess.run(w1))
    print(sess.run(w2))
