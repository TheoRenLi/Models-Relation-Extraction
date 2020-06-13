import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import torch.utils.data as Data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np 
from tqdm import tqdm
from model_and_config.model import L2Regularizer
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
import datetime

# cudnn.benchmark = True
# cudnn.deterministic = True
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

def to_var(x):
    return torch.from_numpy(x)

class Config(object):
    def __init__(self):
        self.data_path = './8类-nostopwords-key1'
        self.use_gpu = True
        self.is_training = True
        self.max_length = 87
        self.pos_num = 148
        self.tag_num = 43
        self.num_class = 10
        self.hidden_size = 256
        self.MLP_dim = 1000
        self.pos_size = 40
        self.tag_size = 30
        self.max_epoch = 15
        self.opt_method = 'Adam'
        self.optimizer = None
        self.learning_rate = 0.001
        # self.weight_decay = 1e-5
        self.drop_prob = 0.5
        self.checkpoint_dir = './checkpoint'
        self.test_result_dir = './test_result'
        self.save_epoch = 1
        self.test_epoch = 1
        self.pretrain_model = None
        self.trainModel = None
        self.testModel = None
        self.batch_size = 32
        self.word_size = 300
        self.epoch_range = None
    def set_data_path(self, data_path):
        self.data_path = data_path
    def set_max_length(self, max_length):
        self.max_length = max_length
    def set_num_class(self, num_class):
        self.num_class = num_class
    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
    def set_MLP_dim(self, MLP_dim):
        self.MLP_dim = MLP_dim
    def set_pos_size(self, pos_size):
        self.pos_size = pos_size
    def set_tag_size(self, tag_size):
        self.tag_size = tag_size
    def set_word_size(self, word_size):
        self.word_size = word_size
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_opt_method(self, opt_method):
        self.opt_method = opt_method
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay
    def set_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob
    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch
    def set_save_epoch(self, save_epoch):
        self.save_epoch = save_epoch
    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model
    def set_is_training(self, is_training):
        self.is_training = is_training
    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range
    
    def load_train_data(self):
        print("Reading training data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_train_word = to_var(np.load(os.path.join(self.data_path, 'train_word.npy')))
        self.data_train_tag = to_var(np.load(os.path.join(self.data_path, 'train_tag.npy')))
        self.data_train_pos1 = to_var(np.load(os.path.join(self.data_path, 'train_pos1.npy')))
        self.data_train_pos2 = to_var(np.load(os.path.join(self.data_path, 'train_pos2.npy')))
        self.train_label = to_var(np.load(os.path.join(self.data_path, 'train_sen_label.npy')))
        self.train_data = torch.cat((self.data_train_word, self.data_train_tag, self.data_train_pos1, self.data_train_pos2), dim=1)
        print("Finish reading.")

    def load_valid_data(self):
        print("Reading validing data...")
        self.data_valid_word = to_var(np.load(os.path.join(self.data_path, 'valid_word.npy')))
        self.data_valid_tag = to_var(np.load(os.path.join(self.data_path, 'valid_tag.npy')))
        self.data_valid_pos1 = to_var(np.load(os.path.join(self.data_path, 'valid_pos1.npy')))
        self.data_valid_pos2 = to_var(np.load(os.path.join(self.data_path, 'valid_pos2.npy')))
        self.valid_label = to_var(np.load(os.path.join(self.data_path, 'valid_sen_label.npy')))
        self.valid_data = torch.cat((self.data_valid_word, self.data_valid_tag, self.data_valid_pos1, self.data_valid_pos2), dim=1)
        print("Finish reading.")
    
    def load_test_data(self):
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_test_word = to_var(np.load(os.path.join(self.data_path, 'test_word.npy')))
        self.data_test_tag = to_var(np.load(os.path.join(self.data_path, 'test_tag.npy')))
        self.data_test_pos1 = to_var(np.load(os.path.join(self.data_path, 'test_pos1.npy')))
        self.data_test_pos2 = to_var(np.load(os.path.join(self.data_path, 'test_pos2.npy')))
        self.test_label = to_var(np.load(os.path.join(self.data_path, 'test_sen_label.npy')))
        self.test_data = torch.cat((self.data_test_word, self.data_test_tag, self.data_test_pos1, self.data_test_pos2), dim=1)
        print("Finish reading.")

    def set_train_model(self, model):
        print('Initializing training model...')
        self.model = model
        self.trainModel = self.model(config=self)
        if torch.cuda.is_available():
            self.trainModel.cuda()
        
        if self.optimizer != None:
            pass
        elif self.optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.trainModel.patameters(), lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.trainModel.patameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.trainModel.parameters() ,lr=self.learning_rate)
        
        self.loss = nn.CrossEntropyLoss()
        print('Finish initializing')
    
    def set_test_model(self, model):
        print('Initializing test model...')
        self.model = model
        self.testModel = self.model(config=self)
        self.testModel.cuda()
        self.testModel.eval()
        print("Finish initializing")

    def dataloader(self, is_type='train'):
        if is_type == 'train':
            train_dataset = Data.TensorDataset(self.train_data, self.train_label)
            train_iter = Data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3)
            return train_iter
        elif is_type == 'valid':
            valid_dataset = Data.TensorDataset(self.valid_data, self.valid_label)
            valid_iter = Data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3)
            return valid_iter
        else:
            test_dataset = Data.TensorDataset(self.test_data, self.test_label)
            test_iter = Data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            return test_iter

    def train_one_step(self, batch, label):
        batch_word = Variable(batch[:, 0:self.max_length].cuda())
        batch_tag = Variable(batch[:, self.max_length:self.max_length * 2].cuda())
        batch_pos1 = Variable(batch[:, self.max_length * 2:self.max_length * 3].cuda())
        batch_pos2 = Variable(batch[:, self.max_length * 3:self.max_length * 4].cuda())
        batch_label = Variable(label.cuda())
        output = self.trainModel(batch_word, batch_tag, batch_pos1, batch_pos2)

        loss = self.loss(output, batch_label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        train_loss, valid_loss = [], []
        train_iter = self.dataloader(is_type='train')
        for epoch in range(self.max_epoch):
            loss_temp = []
            self.trainModel.train()
            print('Epoch ' + str(epoch) + ' starts...')
            for step, (batch, label) in enumerate(train_iter):
                loss = self.train_one_step(batch, label)
                loss_temp.append(loss)
                if step % 20 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("epoch %d step %d time %s | loss: %f\r" % (epoch, step, time_str, loss))
            
            if (epoch + 1) % self.save_epoch == 0:
                print('Epoch ' + str(epoch) + ' has finished')
                print('Saving model...')
                path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
                torch.save(self.trainModel.state_dict(), path) # 注意state_dict()当中的括号
                print('Have saved model to ' + path)
            
            valid_loss_epoch = self.one_epoch_loss()
            train_loss.append(np.mean(loss_temp))
            valid_loss.append(valid_loss_epoch)

        # writer.close()
        self.loss_plot(train_loss, valid_loss)
        

    def one_epoch_loss(self):
        data_loss = []
        pred_label = []
        self.testModel = self.trainModel
        self.testModel.eval()
        valid_iter = self.dataloader(is_type='valid')
        with torch.no_grad():
            for batch, label in valid_iter:
                batch_word = Variable(batch[:, 0:self.max_length].cuda())
                batch_tag = Variable(batch[:, self.max_length:self.max_length * 2].cuda())
                batch_pos1 = Variable(batch[:, self.max_length * 2:self.max_length * 3].cuda())
                batch_pos2 = Variable(batch[:, self.max_length * 3:self.max_length * 4].cuda())
                batch_label = Variable(label.cuda())
                output = self.testModel(batch_word, batch_tag, batch_pos1, batch_pos2)
                loss = self.loss(output, batch_label)
                data_loss.append(loss.item())

                idx = torch.argmax(output, dim=1).cpu().numpy().tolist()
                pred_label.extend(idx)
        test_label = self.valid_label.numpy().tolist()
        report = classification_report(np.array(test_label), np.array(pred_label), output_dict=True)
        macro = self.score(report)
        print("Precision: %f\tRecall: %f\tF1: %f" % (macro['precision'], macro['recall'], macro['f1-score']))

        return np.mean(data_loss)

    def loss_plot(self, train_loss, valid_loss):
        plt.figure('BGRU-CapsNet-loss', figsize=(15, 10))
        plt.plot(train_loss, label='train_loss')
        plt.plot(valid_loss, label='valid_loss')
        x_major_locator = MultipleLocator(1)
        y_major_locator = MultipleLocator(0.05)
        ax = plt.gca()
        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.title("Loss curve")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.test_result_dir,'loss_plot.png'))
        
    def score(self, report):
        macro_new = {}
        macro = report['macro avg']
        other = report['9']
        macro_new['precision'] = (macro['precision'] * 10 - other['precision']) / 9
        macro_new['recall'] = (macro['recall'] * 10 - other['recall']) / 9
        macro_new['f1-score'] = (macro['f1-score'] * 10 - other['f1-score']) / 9
        return macro_new


    def test_one_step(self, batch, label):
        batch_word = Variable(batch[:, 0:self.max_length].cuda())
        batch_tag = Variable(batch[:, self.max_length:self.max_length * 2].cuda())
        batch_pos1 = Variable(batch[:, self.max_length * 2:self.max_length * 3].cuda())
        batch_pos2 = Variable(batch[:, self.max_length * 3:self.max_length * 4].cuda())
        output = self.testModel(batch_word, batch_tag, batch_pos1, batch_pos2)
        output = torch.softmax(output, dim=1)
        return torch.argmax(output, dim=1)
        
    def test_one_epoch(self):
        pred_label = []
        test_iter = self.dataloader(is_type='test')
        for batch, label in tqdm(test_iter):
            idx = self.test_one_step(batch, label)
            idx = idx.cpu().numpy().tolist()
            pred_label.extend(idx)


        test_label = self.test_label.numpy().tolist()
        report_text = classification_report(np.array(test_label), np.array(pred_label))
        print(report_text)
        report = classification_report(np.array(test_label), np.array(pred_label), output_dict=True)
        macro = self.score(report)
        return macro

    def test(self):
        # accuracy = []
        precision = []
        recall = []
        F1 = []
        for epoch in self.epoch_range:
            path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
            if not os.path.exists(path):
                continue
            print("Start testing epoch %d" % (epoch))
            self.testModel.load_state_dict(torch.load(path))

            macro_avg = self.test_one_epoch()
            print("Precision: %f\tRecall: %f\tF1: %f" % (macro_avg['precision'], macro_avg['recall'], macro_avg['f1-score']))
        
            # accuracy.append(acc)
            precision.append(macro_avg['precision'])
            recall.append(macro_avg['recall'])
            F1.append(macro_avg['f1-score'])
            
        print("Storing accuracy-precision-recall-F1...")
        if not os.path.isdir(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        # np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_accuracy.npy'), accuracy)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_precision.npy'), precision)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_recall.npy'), recall)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_F1.npy'), F1)
        print("Finish storing")