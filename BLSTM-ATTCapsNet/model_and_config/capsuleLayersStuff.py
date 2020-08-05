import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable



def squash(x):
    """3维的x的第二维是胶囊数"""
    s_squared_norm = x.pow(2).sum(dim=2, keepdim=True)
    s_norm = s_squared_norm.sqrt()
    scale = s_squared_norm / (1.0 + s_squared_norm) / s_norm
    return scale * x


class AgreementRouting(nn.Module):
    def __init__(self, in_caps, out_caps, n_iterations, num_caps):
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.num_caps = num_caps
        self.b = nn.Parameter(torch.zeros((in_caps, out_caps)), requires_grad=True)

    def forward(self, u_predict, alpha):
        batch_size, in_caps, out_caps, _ = u_predict.size()
        alpha = alpha.expand(alpha.size(0), alpha.size(1), self.num_caps).contiguous()
        alpha = alpha.view(batch_size, -1, 1, 1)

        c = F.softmax(self.b, dim=1)
        s = (c.unsqueeze(2) * alpha * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, in_caps, out_caps))
            for _ in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(dim=-1)

                c = F.softmax(b_batch, dim=2).unsqueeze(3)
                s = (c * alpha * u_predict).sum(dim=1)
                v = squash(s)
        return v

class CapsLayer(nn.Module):
    def __init__(self, in_caps, in_dim, out_caps, out_dim, routing_module):
        super(CapsLayer, self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        # 权重共享
        self.weights = nn.Parameter(torch.randn(self.in_dim, self.out_caps * self.out_dim))
        # 权重不共享
        # self.weights = nn.Parameter(torch.randn(self.in_caps, self.in_dim, self.out_caps * self.out_dim))
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
    
    def forward(self, caps_output, alpha):
        caps_output = caps_output.unsqueeze(2)
        u_predict = caps_output.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.in_caps, self.out_caps, self.out_dim)
        v = self.routing_module(u_predict, alpha)
        return v

class PrimaryCapsLayer(nn.Module):
    def __init__(self, in_caps, in_dim):
        super(PrimaryCapsLayer, self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, he):
        alpha = self.sigmoid(x @ he)
        out = x.view(x.size(0), x.size(1), self.in_caps, self.in_dim)
        out = out.contiguous()
        out = out.view(out.size(0), -1, out.size(3))
        out = squash(out)
        return out, alpha


class MarginLoss(nn.Module):
    def __init__(self, gamma=0.4, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.B = nn.Parameter(torch.Tensor([0.5]))

    def forward(self, y_pred, y_true):
        t = torch.zeros(y_pred.size()).long()
        if y_true.is_cuda:
            t = t.cuda()
        t = t.scatter_(1, y_true.data.view(-1, 1), 1)
        
        # y_true is the code of one-hot
        y_true = Variable(t)
        losses = y_true.float() * F.relu((self.B + self.gamma) - y_pred).pow(2) + self.lambda_ * (1. - y_true.float()) * F.relu(y_pred - (self.B - self.gamma)).pow(2)
        return losses.sum(dim=1).mean()


# class MarginLoss(nn.Module):
#     def __init__(self, m_pos=0.9, m_neg=0.1, lambda_=0.5):
#         super(MarginLoss, self).__init__()
#         self.m_pos = m_pos
#         self.m_neg = m_neg
#         self.lambda_ = lambda_

#     def forward(self, y_pred, y_true):
#         t = torch.zeros(y_pred.size()).long()
#         if y_true.is_cuda:
#             t = t.cuda()
#         t = t.scatter_(1, y_true.data.view(-1, 1), 1)
        
#         # y_true is the code of one-hot
#         y_true = Variable(t)
#         losses = y_true.float() * F.relu(self.m_pos - y_pred).pow(2) + self.lambda_ * (1. - y_true.float()) * F.relu(y_pred - self.m_neg).pow(2)
#         return losses.sum(dim=1).mean()


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
                reg_loss_func += 0.5 * self.lambda_reg * L2Regularizer.__add_l2(var=model_param_value)
        return reg_loss_func

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum().sqrt()