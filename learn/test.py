sign_dict = {
    "f1_score": 1,
    "f2_score": 1,
    "auc": 1, "multi_auc": 1, "acc": 1, "binary_acc": 1,
    "mse": -1, "ber": -1, "log_loss": -1, "correlation": 1
}
require_prob = {
    name: False for name in sign_dict
}

for key in require_prob:
    print('key=%s, value=%s' % (key, require_prob[key]))