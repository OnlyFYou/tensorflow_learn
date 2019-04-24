import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 100  # 第一层隐藏层节点数
BATCH_SIZE = 100  # 一个训练 batch 中的训练数据个数。数字越小时，训练过程越接近随机梯度下降；数字越大时，训练越接近梯度下降
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里定义了一个使用 ReLU 激活函数的三层全连接神经网络。
# 通过加入隐藏层实现了多层网络结构， 通过 ReLU 激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值得类，这样方便在
# 测试时使用滑动平均模型。
def inference(input_tensor, avg_class, reuse=False):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if not avg_class:
        with tf.variable_scope('layer1', reuse=reuse):
            weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
            # 计算隐藏层的前向传播结果，这里使用了 ReLU 激活函数
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
            # 计算输出层的前向传播结果。因为在计算损失函数时会一并计算 softmax 函数，所以这里不需要加入激活函数。
            # 而且不加入 softmax 不会影响预测结果。因为预测时使用的是不同类别对应节点输出值的相对大小，有没有
            # softmax 层对最后分类结果的计算没有影响。于是在计算整个神经网络的前向传播时可以不加入最后的 softmax 层。
            return  tf.matmul(layer1, weights) + biases
    else:
        with tf.variable_scope('layer1', reuse=reuse):
            weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
            # 生成输出层的参数
            # 首先使用 avg_class.average 函数来计算得出变量的滑动平均值，然后再计算相应的神经网络前向传播结果。
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))
        with tf.variable_scope('layer2', reuse=reuse):
            weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
            return tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases)


# 训练模型的过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为 None，所以函数不会使用参数的滑动平均值
    y = inference(x, None, False)
    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练的变量（trainable=False）。
    # 在使用 Tensorflow 训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。在第4章中介绍过给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如 global_step ）就不需要了。
    # tf.trainable_variables 返回的就是图上集合 GraphKeys.TRAINABLE_VARIABLES 中的元素。
    # 这个集合的元素就是所有没有指定 trainable=False 的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果。第4章中介绍过滑动品均不会改变变量本身的取值，而是会维护一个影子变量来记录其滑动平均值。
    # 所以当需要使用这个滑动平均值时，需要明确调用 average 函数
    average_y = inference(x, variable_averages, True)

    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了 Tensorflow 中提供的 sparse_softmax_cross_entropy_with_logits
    # 函数来计算交叉熵。当分类问题只用一个正确答案时，可以使用这个函数来加速交叉熵的计算。 MNIST 问题的图片中只包含 0~9 中的一个数字，
    # 所以可以使用这个函数来计算交叉熵损失。这个函数的第一个参数是神经网络不包含 softmax 层的前向传播结果，第二个是训练数据的正确答案。
    # 因为标准答案是一个长度为 10 的一个数组，而该函数需要提供的是一个正确答案的数字，所以需要使用 tf.argmax 函数来得到正确答案对应的类别编号。
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前 batch 中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    with tf.variable_scope('layer1', reuse=True):
        regu1 = regularizer(tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE]))
    with tf.variable_scope('layer2', reuse=True):
        regu2 = regularizer(tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE]))
    # 计算模型的正则化损失。一般只计算神经网络边上权重的正则化损失，而不使用偏置项。
    regularization = regu1 + regu2
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率，随着迭代的进行，更新变量时使用的学习率在这个基础上递减
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY)  # 学习率衰减速度

    # 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数。注意这里损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 在训练神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又要更新每一个参数的轮滑平均值。
    # 为了一次完成多个操作，Tensorflow 提供了 tf.control_dependencies 和 tf.group 两种机制。下面两行程序和
    # tf.group(train_step, variables_averages_op) 是等价的
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验使用了滑动平均模型的神经网络前向传播结果是否正确。tf.argmax(y_, 1) 计算每一个样例的预测答案。
    # 相等，如果相等返回 True，否则返回 False。
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 这个运算首先将一个布尔型的数值转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        # 准备验证数据。一般在神经网络的训练过程中会通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 准备测试数据。在真实的应用中，这部分数据在训练时是不可见的，这个数据只是作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                # 计算滑动平均模型在验证数据上的结果。因为 MNIST 数据集比较小，所以一次可以处理所有的验证数据。
                # 为了计算方便，本样例程序没有将验证数据划分为更小的 batch。当神经网络模型比较复杂或验证数据比较大时，
                # 太大的 batch 会导致计算时间过长甚至发生内存溢出的错误。
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))


def main(arvg=None):
    mnist = input_data.read_data_sets("/path/to/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
