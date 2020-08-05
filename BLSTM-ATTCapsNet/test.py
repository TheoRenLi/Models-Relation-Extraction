from model_and_config import config, models
import os
import argparse


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='LSTMCapsNet', help='name of the model')
args = parser.parse_args()
model = {'LSTMCapsNet': models.LSTMCapsNet}

con = config.Config()
con.load_test_data()
con.set_test_model(model[args.model_name])
a = list(range(2, 15))
con.set_epoch_range(a)
con.test()
