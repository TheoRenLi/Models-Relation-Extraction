from model_and_config import config
from model_and_config import  model
import numpy as np
import argparse

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='LSTM', help='name of the model')
args = parser.parse_args()
model = {'LSTM': model.LSTM}

con = config.Config()
con.load_test_data()
con.set_test_model(model[args.model_name])
a = list(range(0, 25))
con.set_epoch_range(a)
con.test() 