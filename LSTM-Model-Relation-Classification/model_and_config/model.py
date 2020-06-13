import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np



class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.input_size = self.config.word_size + self.config.tag_size + 2 * self.config.pos_size
        self.hidden_dim = self.config.hidden_size
        self.linear_dim = self.input_size * 2 + self.hidden_dim * 4 * 2

        self.embed = Embed_layer(self.config)
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(self.linear_dim, self.config.MLP_dim),
            nn.ReLU(),
            nn.Linear(self.config.MLP_dim, self.config.num_class)
        )
        self.dropout = nn.Dropout(self.config.drop_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
        for name, param in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, word, tag, pos1, pos2):
        x = self.embed(word, tag, pos1, pos2)
        x = self.dropout(x)
        batch = range(x.size(0))
        index_e1, index_e2 = self.getEntityIndex(pos1, pos2)
        e1 = x[batch, index_e1, :]
        e2 = x[batch, index_e2, :]
        x, (_, _) = self.lstm(x)
        x = self.dropout1(x)
        fe1 = x[batch, index_e1, :]
        fe2 = x[batch, index_e2, :]
        x = self.getSenFeature(x, index_e1, index_e2)
        x = torch.cat([x, e1, fe1, e2, fe2], dim=1)
        logits = self.linear(x)
        return logits


    def getSenFeature(self, x, index_e1, index_e2):
        out = []
        for i in range(len(index_e2)):
            try:
                m1, _ = torch.max(x[i, :index_e2[i], :], dim=0)
            except RuntimeError as e:
                m1 = x[i, index_e2[i], :]
            m2, _ = torch.max(x[i, index_e1[i]:, :], dim=0)
            out.append(torch.cat([m1, m2]).unsqueeze(0))
        return torch.cat(out, dim=0)

    def getEntityIndex(self, pos1, pos2):
        index_e1 = []
        index_e2 = []
        for i in range(pos1.size(0)):
            index_e1.append(pos1[i].cpu().numpy().tolist().index(68))
            index_e2.append(pos2[i].cpu().numpy().tolist().index(68))
        return index_e1, index_e2



class Embed_layer(nn.Module):
    def __init__(self, config):
        super(Embed_layer, self).__init__()
        self.config = config
        
        self.word_embedding = nn.Embedding(self.config.data_word_vec.shape[0], self.config.data_word_vec.shape[1])
        self.tag_embedding = nn.Embedding(self.config.tag_num, self.config.tag_size, padding_idx=0)
        self.pos1_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(self.config.data_word_vec), requires_grad=False)
        nn.init.xavier_uniform_(self.tag_embedding.weight)
        if self.tag_embedding.padding_idx is not None:
            self.tag_embedding.weight.data[self.tag_embedding.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        if self.pos1_embedding.padding_idx is not None:
            self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        if self.pos2_embedding.padding_idx is not None:
            self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)

    def forward(self, word, tag, pos1, pos2):
        word_embed = self.word_embedding(word)
        tag_embed = self.tag_embedding(tag)
        pos1_embed = self.pos1_embedding(pos1)
        pos2_embed = self.pos2_embedding(pos2)
        embed = torch.cat([word_embed, tag_embed, pos1_embed, pos2_embed], dim=2)
        return embed



class L2Regularizer:
    def __init__(self, model, lambda_reg, name_reg):
        super(L2Regularizer, self).__init__()
        self.model = model
        self.lambda_reg = lambda_reg
        self.name_reg = name_reg
    
    def regularized_param(self, param_weights, reg_loss_func):
        reg_loss_func += self.lambda_reg * L2Regularizer.__add_l2(var=param_weights)
        return reg_loss_func

    def regularize_all_param(self, reg_loss_func):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name in self.name_reg:
                reg_loss_func += self.lambda_reg * L2Regularizer.__add_l2(var=model_param_value)
        return reg_loss_func

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum().sqrt()