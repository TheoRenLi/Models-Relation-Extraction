from model_and_config import config
from model_and_config import model
import argparse
import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='LSTM', help='name of the model')
    parser.add_argument('data_path', type=str, default='./8类-nostopwords-key1', help='path of data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch of per training')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size of LSTM')
    parser.add_argument('--MLP_dim', type=int, default=1000, help='num. of unit of MLP')
    parser.add_argument('--pos_size', type=int, default=40, help='size of position embedding')
    parser.add_argument('--tag_size', type=int, default=30, help='size of pos tag embedding')
    parser.add_argument('--max_epoch', type=int, default=25, help='num. of epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--drop_prob', type=float, default=0.5, help='dropout probability')
    args = parser.parse_args()
    models = {'LSTM': model.LSTM}

    con = config.Config()
    con.set_data_path(args.data_path)
    con.set_max_epoch(args.max_epoch)
    con.set_batch_size(args.batch_size)
    con.set_hidden_size(args.hidden_size)
    con.set_MLP_dim(args.MLP_dim)
    con.set_pos_size(args.pos_size)
    con.set_tag_size(args.tag_size)
    con.set_learning_rate(args.lr)
    con.set_drop_prob(args.drop_prob)
    con.load_train_data()
    con.load_valid_data()
    con.set_train_model(models[args.model_name])
    con.train()


if __name__ == "__main__":
    main_config()
