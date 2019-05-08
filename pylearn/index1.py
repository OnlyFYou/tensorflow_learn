import numpy as np
with open('Data.txt', 'r') as file:
    data = np.array([line.strip().split(',') for line in file], dtype=np.float32)
length = len(data)
# 训练集和交叉验证集
n_train, n_cv = int(0.7*length), int(0.15*length)
# idx=100
idx = np.random.permutation(length)
train_idx, cv_idx = idx[:n_train], idx[n_train:n_train+n_cv]
test_idx = idx[n_train+n_cv:]
train, cv, test = data[train_idx], data[cv_idx], data[test_idx]
