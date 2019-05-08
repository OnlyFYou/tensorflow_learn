import numpy as np


class Toolbox:
    @staticmethod
    def get_data(file, sep=' ', include_header=False):
        print('Fetching data')
        data = [
            [elem if elem else 'nan' for elem in line.strip().split(sep)]
            for line in file
        ]
        if include_header:
            return data[1:]
        return data

    # 将y转为ont-hot representation
    @staticmethod
    def get_one_hot(y, n_classes):
        if y is None:
            return
        ont_hot = np.zeros([len(y), n_classes])
        ont_hot[range(len(ont_hot)), np.asarray(y, np.int)] = 1
        return ont_hot