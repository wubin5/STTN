# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:13:06 2020

@author: wb
"""
import torch
import torch.nn as nn

class One_hot_encoder(nn.Module):
    def __init__(self, embed_size, time_num=288):
        super(One_hot_encoder, self).__init__()
        self.time_num = time_num
        self.I = nn.Parameter(torch.eye(self.time_num, self.time_num, requires_grad=True))
        self.onehot_Linear = nn.Linear(time_num, embed_size)

    def forward(self, i, N=25, T=12):
    
        if i%self.time_num+T > self.time_num :
            o1 = self.I[i%self.time_num : , : ]
            o2 = self.I[0 : (i+T)%self.time_num, : ]
            onehot = torch.cat((o1, o2), 0)
        else:        
            onehot = self.I[i%self.time_num: i%self.time_num+T, : ]
        
        #onehot = onehot.repeat(N, 1, 1)   
        onehot = onehot.expand(N, T, self.time_num)
        onehot = self.onehot_Linear(onehot)
        return onehot
'''
def one_hot_function(i, time_num=288, N=25, T=12):
    
    I = torch.eye(time_num, time_num)
    
    if i%time_num+T > time_num :
        o1 = I[i%time_num : , : ]
        o2 = I[0 : (i+T)%time_num, : ]
        onehot = torch.cat((o1, o2), 0)
    else:        
        onehot = I[i%time_num: i%time_num+T, : ]
        
    #onehot = onehot.repeat(N, 1, 1)   
    onehot = onehot.expand(N, T, time_num)
    
    return onehot'''
  

