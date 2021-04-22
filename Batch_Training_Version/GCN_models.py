import torch.nn as nn
import torch.nn.functional as F
from layers import cheb_conv
import numpy as np


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, cheb_K, dropout):
        super(GCN, self).__init__()

        self.gc1 = cheb_conv(nfeat, nhid, adj, cheb_K)
        self.gc2 = cheb_conv(nhid, nclass, adj, cheb_K)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x)
        return F.log_softmax(x, dim=1)
