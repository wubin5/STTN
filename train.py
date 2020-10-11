# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:25:49 2020

@author: wb
"""
import torch
import torch.nn as nn
from ST_Transformer import STTransformer
import pandas as pd
import numpy as np


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    '''
    x1 = torch.tensor(                       # x shape[C, N, T] 
            [
             [
             [6.5,  5, 6, 4, 3, 9, 5, 2, 0], 
             [4, 8, 7, 3, 4, 5, 6.5, 7, 2],
             [5, 6, 8, 9.1, 21, 4, 4, 6,20],
             [2, 6, 8, 1, 3, 0, 2.2, 2, 5]
             ]
            ]
    ).to(device)
    
    Aa = torch.tensor([
            [1,0,1,0],
            [0,1,0,1],
            [2,0,1,0],
            [1,2,0,1.]
            ]
    ).to(device)        #邻接矩阵adj'''

    days = 10
    val_days = 1    
    
    train_num = 288*days
    val_num = 288*val_days
    row_num = train_num + val_num

    v = pd.read_csv("PEMSD7/V_25.csv", nrows = row_num, header= -1)
    A = pd.read_csv("PEMSD7/W_25.csv", header= -1)
    

    A = np.array(A)
    A = torch.tensor(A, dtype=torch.float32)
       
    v = np.array(v)
    v = v.T
    v = torch.tensor(v, dtype=torch.float32)
    
    
    '''
    x = v[:, 0:12]
    y = v[:, 12:]
    
    x = x.unsqueeze(0)
    in_channels = v.shape[0]'''
    
    in_channels=1
    embed_size=64
    time_num = 288  #1天时间间隔数
    num_layers=1
    T_dim=12
    output_T_dim=3
    heads=1
    
    
    model = STTransformer(in_channels, embed_size, time_num, num_layers, T_dim, output_T_dim, heads)   
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)  #小数点后8位
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()                             #论文要求
    
    
    for i in range(train_num - 15):
        x = v[:, i:i+12]
        x = x.unsqueeze(0)
        y = v[:, i+12:i+15]
        
        out = model(x, A, i)
        loss = criterion(out, y ) 
        
        if i%100 == 0:
            #print("out", out)
            print("MAE loss:", loss)
        
        #常规操作
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
    
    #print("输出形状", out.shape)
    torch.save(model, "model.pth")
    
    
    
    
    
    
    
    
    
    