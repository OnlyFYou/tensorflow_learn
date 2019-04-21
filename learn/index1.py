import tensorflow as tf
'''
正则化损失 在损失安函数中加入刻画模型复杂度的指标 用来防止  过拟合问题 
J(A) + aR(w) : J(A):损失函数  R(w):模型的复杂程度 a:模型复杂损失在总损失中的比例
'''
#获取一层神经网络边上的权重，并将这个权重的L2正则化损失函数加入‘losses’集合中,shape表示正则化项的权重，lambd需要计算正则化损失的参数
def get_weight(shape, lambd):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(lambd),(var))
    return var

x = tf.placeholder(tf.float32, shape = (None, 2))
y_ = tf.placeholder(tf.float32, shape = (None, 1))
batch_size = 8

#每一层网络中的节点数
layer_dimension = [2, 10, 10, 10, 1]

#神经网络的总层数
n_layer = len(layer_dimension)

#当前网络层深
cur_layer=x

#当前层节点个数
in_dimension = layer_dimension[0]

#通过一个循环生成五层全连接的神经网络结构
for i in range(1, n_layer):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    #使用ReLU激活函数
    cur_layer = tf.nn.rule(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection('losses', mes_loss)

#损失函数总量
loss = tf.add_n(tf.get_collection('losses'))