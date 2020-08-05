import torch
from torch import nn
from model_and_config.capsuleLayersStuff import *



class LSTMCapsNet(nn.Module):
    
    def __init__(self, config):
        super(LSTMCapsNet, self).__init__()
        self.config = config
        self.input_size = self.config.word_size + 2 * self.config.pos_size + self.config.tag_size
        self.hidden_dim = self.config.hidden_size
        self.num_caps = self.config.num_caps
        
        self.embedLayer = EmbedLayer(self.config)
        self.blstm = nn.LSTM(self.input_size, self.hidden_dim, batch_first=True, bidirectional=True)

        self.primaryCaps = PrimaryCapsLayer(in_caps=self.num_caps, in_dim=16)
        self.numPCaps = self.num_caps * self.config.max_length
        routingModule = AgreementRouting(self.numPCaps, self.config.num_class, self.config.routing_iter, self.num_caps)
        self.digitCaps = CapsLayer(self.numPCaps, 16, self.config.num_class, 16, routingModule)
        self.init_weights()
        self.dropout = nn.Dropout(self.config.drop_prob)
        
    def forward(self, word, tag, pos1, pos2):
        x = self.embedLayer(word, tag, pos1, pos2)
        x = self.dropout(x)
        x, _ = self.blstm(x)
        x = x[:, :, 0:self.hidden_dim] + x[:, :, self.hidden_dim:self.hidden_dim*2]
        he = self.getEntityFeature(x, pos1, pos2)

        x, alpha = self.primaryCaps(x, he)
        x = self.digitCaps(x, alpha)
        probs = x.pow(2).sum(dim=2).sqrt()
        return probs

    def getEntityFeature(self, x, pos1, pos2):
        batch = range(x.size(0))
        index_e1, index_e2 = list(), list() 
        for i in batch:
            index_e1.append(pos1[i].cpu().numpy().tolist().index(68))
            index_e2.append(pos2[i].cpu().numpy().tolist().index(68))
        return (x[batch, index_e1, :] + x[batch, index_e2, :]).unsqueeze(2)
    
    def init_weights(self):
        for name, param in self.blstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)



class EmbedLayer(nn.Module):
    def __init__(self, config):
        super(EmbedLayer, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.data_word_vec.shape[0], config.data_word_vec.shape[1])
        self.tag_embedding = nn.Embedding(config.tag_num, config.tag_size, padding_idx=0)
        self.pos1_embedding = nn.Embedding(config.pos_num, config.pos_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(config.pos_num, config.pos_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.config.data_word_vec))
        self.word_embedding.weight.requires_grad = False
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