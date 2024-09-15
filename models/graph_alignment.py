import torch 
import torch.nn as nn
import math
from torch.nn import Parameter
import numpy as np
import torch.nn.functional as F
import random

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


class GraphConvDiscriminator(nn.Module):
    def __init__(self, in_features, in_dim, out_dim, drop_rat, n, pool_shape):
        super(GraphConvDiscriminator, self).__init__()
        
        self.in_features = in_features
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.drop_rat = drop_rat
        self.n = n
        self.pool_shape = pool_shape

        self.gc1 = GraphConvolution(in_features=self.in_features, out_features=self.in_features)
        self.gc1_relu = nn.ReLU()
        self.gc2 = GraphConvolution(in_features=self.in_features, out_features=self.in_features)
        self.gc2_relu = nn.ReLU()
        self.gc3 = GraphConvolution(in_features=self.in_features, out_features=self.in_features)
        self.gc3_relu= nn.ReLU()

        self.logit = GraphConvolution(in_features=self.in_features, out_features=1)
        self.sigmoid = nn.Sigmoid()

        self.lin = nn.Linear(self.in_dim, self.out_dim)
        self.lin_relu = nn.ReLU()

        self.dropout = nn.Dropout(p=self.drop_rat)
        self.pool = nn.AdaptiveAvgPool2d(pool_shape)

        self.ind1 = nn.Parameter(torch.randn(self.n), requires_grad= True)
        self.ind2 = nn.Parameter(torch.randn(self.n), requires_grad= True)

        self.initialize_weights()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0


    def initialize_weights(self):
        self.gc1.initialize_weights()
        self.gc2.initialize_weights()
        self.gc3.initialize_weights()


    def forward(self, x, x_s=None):
        adj1_idx = torch.topk(F.gumbel_softmax(self.ind1, hard=True), 1).indices
        adj2_idx = torch.topk(F.gumbel_softmax(self.ind2, hard=True), 1).indices
        adj1_idx, adj2_idx = adj1_idx.item(), adj2_idx.item()
        
        adj1 = self.lin_relu(self.lin(x[adj1_idx].contiguous().view(-1).unsqueeze(0)))
        adj1 = adj1.squeeze(0).contiguous().view(x.size(2),x.size(2))
        if x_s is not None:
            adj2 = self.lin_relu(self.lin(x_s[adj2_idx].contiguous().view(-1).unsqueeze(0)))
            adj2 = adj2.squeeze(0).contiguous().view(x_s.size(2),x_s.size(2))
        else:
            adj2 = self.lin_relu(self.lin(x[adj2_idx].contiguous().view(-1).unsqueeze(0)))
            adj2 = adj2.squeeze(0).contiguous().view(x.size(2),x.size(2))

        adj1_norm = F.softmax(adj1)
        adj2_norm = F.softmax(adj2)

        adj = F.cosine_similarity(adj1_norm.unsqueeze(0), adj2_norm.unsqueeze(2), dim=2).detach()

        if self.training:
            self.iter_num += 1

        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        if x_s is not None:
            x_s = x * 1.0
        x = x * 1.0

        if x_s is not None and self.training and x_s.requires_grad:
            x_s.register_hook(grl_hook(coeff))
        if self.training and x.requires_grad:
            x.register_hook(grl_hook(coeff))

        if x_s is not None:
            gc_x = self.gc1_relu(self.gc1(torch.cat((x, x_s), dim=0), adj))
        else:
            gc_x = self.gc1_relu(self.gc1(x, adj))
        
        gc_x = self.dropout(self.pool(gc_x))

        gc_x = self.gc2_relu(self.gc2(gc_x, adj))
        gc_x = self.gc3_relu(self.gc3(gc_x, adj))

        gc_y = self.logit(gc_x, adj)

        out = self.sigmoid(gc_y)
        out = out.permute(0,3,2,1)
            
        gc_y = F.softmax(gc_y)
        gc_y = gc_y.permute(0,3,2,1)

        return out, gc_y