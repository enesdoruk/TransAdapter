import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,sigmoid=False,reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce

    def forward(self, inputs, targets):
        P = F.sigmoid(inputs)

        if targets[0][0][0] == 0:
            probs = 1 - P
            log_p = probs.log()
            batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        if targets[0][0][0] == 1:
            probs = P
            log_p = probs.log()
            batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
    
        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss




def adv_local(features, features_s=None, ad_net=None, is_source=False):
    if features_s is not None:
        feats = torch.cat((features, features_s), dim=1)
    else:
        feats = features
    out, att = ad_net(feats)
    out, att = out.squeeze(1), att.squeeze(1)
    batch_size = out.size(0)
    num_heads = out.size(1)
    seq_len = out.size(2)

    if is_source:
        label = torch.from_numpy(np.array([[[1]*seq_len]*num_heads] * batch_size)).float().to(features.device)
    else:
        label = torch.from_numpy(np.array([[[0]*seq_len]*num_heads] * batch_size)).float().to(features.device)
       
    return out, torch.nn.BCELoss()(out, label), att



    
def adv_global(loss, x, x_t, ad_net):
    x = x.reshape(x.size(0), int(x.size(1)**0.5), int(x.size(1)**0.5), x.size(2))
    x_t = x_t.reshape(x_t.size(0), int(x_t.size(1)**0.5), int(x_t.size(1)**0.5), x_t.size(2))
    features = torch.cat((x, x_t), 0)
    
    out, att = ad_net(features)
    out, att = out.squeeze(1), att.squeeze(1)
    num_heads = out.size(1)
    seq_len = out.size(2)

    batch_size = out.size(0) // 2

    src_target =  torch.from_numpy(np.array([[[1]*seq_len]*num_heads] * batch_size)).float().to(features.device)
    trgt_target =  torch.from_numpy(np.array([[[0]*seq_len]*num_heads] * batch_size)).float().to(features.device)

    loss_src = loss(out[:batch_size], src_target) 
    loss_trgt = loss(out[batch_size:], trgt_target)

    return (loss_src + loss_trgt) / 2