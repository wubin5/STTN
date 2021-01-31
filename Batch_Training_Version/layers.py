import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import scaled_Laplacian, cheb_polynomial
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    
class cheb_conv(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''
    def __init__(self, in_features, out_features, adj, K, bias=True):
#     def __init__(self, nfeat, nhid, nclass, dropout, K):
#     def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.DEVICE = 'cuda:0'
        self.K = K
        adj = np.array(adj.cpu())
        L_tilde = scaled_Laplacian(adj)
        self.cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(self.DEVICE) for i in cheb_polynomial(L_tilde, K)]
        
        
        self.in_channels = in_features
        self.out_channels = out_features
        
        self.Theta = nn.ParameterList([nn.Parameter(torch.randn(self.in_channels, self.out_channels).to(self.DEVICE)) for _ in range(K)])

    def forward(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (B, N, C] --> (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''
#         x = x.permute(0, 1, 3, 2)
        batch_size, num_of_vertices, in_channels = x.shape

           

        graph_signal = x

        output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

        for k in range(self.K):

            T_k = self.cheb_polynomials[k]  # (N,N)

            theta_k = self.Theta[k]  # (in_channel, out_channel)

            rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1) # （b, F_in, N) * (N, N) --> (b, F_in, N) --> (b, N, F_in)

            output = output + rhs.matmul(theta_k) # (b, N, F_in) * (F_in, F_out) --> (b, N, F_out) 


        
        result = F.relu(output)

        return result