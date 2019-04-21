import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#输入层节点数：图片的特征向量的维度
INPUT_NODE = 784
#分类的类别数量
OUT_NODE = 10

LAYER1_NODE = 500
#一个训练batch中的数据量
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8 #基础学习率
LEARNING_RATE_DECAY = 0.99 #学习率的衰减率 一般接近1

REGUALRIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数

TRAINING_STEPS = 30000 #训练轮数
MOVING_AVERAGE_DEACY = 0.99 #滑动平均衰减率

#定义一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        #没有滑动平均类，
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2))+ avg_class.average(biases2)


#训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None,INPUT_NODE], name = "x-input")
    y_ = tf.placeholder(tf.float32, [None, OUT_NODE], name = "y-input")
    #生成神经网络参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUT_NODE]))

    #计算前向传播结果
    y = inference(x,None,weights1,biases1,weights2,biases2)

    #定义存储训练轮数的变量。这里变量不需要计算滑动平均值，
    # 所以指定这个变量为不可训练的变量，在使用tensorflow训练神经网络时，一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0,trainable=False)
    #初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DEACY, global_step)
    #在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #计算使用滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    #计算交叉熵作为损失，使用softmax函数计算交叉熵，当分类问题中只有一个正确答案，可以使用这个函数 argmax函数计算最大值 0：按列 1：按行
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGUALRIZATION_RATE)
    #计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，不适用偏置项
    regularzation = regularizer(weights1)+regularizer(weights2)
    loss = cross_entropy_mean + regularzation

    #指定指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples / BATCH_SIZE,LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
