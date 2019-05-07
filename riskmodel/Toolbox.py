import numpy as np
class Toolbox:
    @staticmethod
    def get_data(file, sep=" ", inclued_header=False):
        print('fetching data')
        data = [
            [elem if elem else 'nan' for elem in line.strip().split(sep)] for line in file
        ]
        if inclued_header:
            return data[1:]
        return data

    # 定义能够将y转化成ont_hot，representation的方法
    @staticmethod
    def get_ont_hot(y, n_classes):
        if y is None:
            return
        one_hot = np.zeros([len(y), n_classes])
        one_hot[range(len(one_hot)), np.asarray(y, np.int)] = 1
        return one_hot