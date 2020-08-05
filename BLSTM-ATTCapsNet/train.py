from model_and_config import config
from model_and_config import model
import argparse
import torch
import os
import json
import numpy as np 
import random

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
path = './test_result/'

def seed_determi():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main_config(name, params):
    param_ = list()
    for param in params:
        print("{} value is: {}".format(name, param))
        seed_determi()
        dict_ = dict()
        models = {'lstmcapsnet': model.LSTMCapsNet}

        con = config.Config()
        con.set_max_epoch(25)
        con.set_learning_rate(0.004)
        con.set_batch_size(32)
        con.set_pos_size(30)
        con.set_tag_size(30)
        con.set_hidden_size(256)
        con.set_num_caps()
        con.set_drop_prob(0.5)
        con.set_weight_decay(0.0)
        con.load_train_data()
        con.load_test_data()
        con.set_train_model(models['lstmcapsnet'])
        best_F1 = con.train()
        dict_[name] = param
        dict_['BestF1'] = best_F1
        param_.append(dict_)
        torch.cuda.empty_cache()
    with open(os.path.join(path, name + '.json'), 'a+') as f:
        json.dump(param_, f, indent=4)
        f.close()


if __name__ == "__main__":
    name = 'weight_decay'
    params = [1e-5]
    main_config(name, params)
