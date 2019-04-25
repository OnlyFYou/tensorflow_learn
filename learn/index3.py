import tensorflow as tf

'''
    滑动平均模型， 在采用梯度下降算法训练神经网络时，使用滑动平均模型可以在一定程度上提高最终模型在测试数据上的表现   
    tf.train.ExxponentialMovingAverage(0.99, step)
'''

# 定义一个初始变量 用户计算滑动平均
v1 = tf.Variable(0, dtype=tf.float32)

# step变量模拟神经网络迭代的轮数，可以用于控制衰减率
step = tf.Variable(0, trainable=False)
# 滑动平均类
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作，这个列表中的变量都会更新
maintain_averages_op = ema.apply([v1])
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average(v1) 获取滑动平均之后变量的取值。在初始化之后变量v1的值和v1的滑动平均都是0
    print(sess.run([v1, ema.average(v1)]))
    # v1 --> 5  衰减率 min{0.99,(1+step)/(10+step) = 0.1} = 0.1
    sess.run(tf.assign(v1, 5))
    # v1的滑动平均值 0.1* 0 +0.9 *5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
