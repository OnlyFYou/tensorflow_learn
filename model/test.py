from Util import Activations, Losses
from Train import TrainModel

data = {
    "train_file_ary": ['C:\\Workspace\\tensorflow_learn\\model\\data\\test.csv'],
    "test_file_ary": ['C:\\Workspace\\tensorflow_learn\\model\\data\\train.csv'],
    "total_steps": 5000,
    "model_path": "/model/to/model",
    "model_log": "/model/to/log",
    "n_node_ary": [14, 200, 2],
    "feature_size": 14,
    "act": Activations.relu,
    "regu_bool": False,
    "loss_fun": Losses.cross_entropy,
    "train_data_size": 4833,
    "batch_size": 100,
    "test_data_size": 50

}
TrainModel.train_start(data)
