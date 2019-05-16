from Util import Activations, Losses
from Train import TrainModel

data = {
    "train_file_ary": ['C:\\Workspace\\score_model\\model\\data\\train.csv'],
    "test_file_ary": ['C:\\Workspace\\score_model\\model\\data\\test.csv'],
    "total_steps": 5000,
    "model_path": "/model/to/model",
    "model_log": "/model/to/log",
    "n_node_ary": [14, 200, 150, 2],
    "feature_size": 15,
    "act": Activations.relu,
    "regu_bool": False,
    "loss_fun": Losses.cross_entropy,
    "train_data_size": 4600,
    "batch_size": 50,
    "test_data_size": 285

}
TrainModel.train_start(data)
