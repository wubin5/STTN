# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 22:20:00 2020

@author: wb
"""

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

def MAE(x, y):   #zi自己做MAE
    out = torch.abs(x-y)
    return out.mean(dim=0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    days = 10
    val_days = 2    #需要验证天数
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
    
    
    in_channels=1
    embed_size=64
    time_num = 288
    num_layers=1
    T_dim=12
    output_T_dim=3
    heads=2
       
    #model = STTransformer(in_channels, embed_size, time_num, num_layers, T_dim, output_T_dim, heads)      
    model = torch.load('model.pth')
    criterion1 = nn.L1Loss()   #MAE
    criterion3 = nn.MSELoss()  #RMSE
    
    
    for i in range( train_num , row_num-15  ):
        x = v[:, i:i+12]
        x = x.unsqueeze(0)
        y = v[:, i+12:i+15]
        
        out = model(x, A, i)
        
        #out=out.T
        #y=y.T
        
        loss1 = criterion1(out, y ) 
        loss2 = MAE(out, y)
        loss3 = torch.sqrt(criterion3(out, y ) )
        if i%100 == 0:
            #print("out", out)
            print("MAE  loss", loss1)
            print("Loss2:", loss2)
            print("RMSE loss", loss3)
        

        
    
    #print(out)
    #print("输出形状", out.shape)

    
    
    
    
    
    
    
    
    
    