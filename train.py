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

    days = 10       #选择训练的天数
    val_days = 1    #选择验证的天数
    
    train_num = 288*days
    val_num = 288*val_days
    row_num = train_num + val_num

    v = pd.read_csv("PEMSD7/V_25.csv", nrows = row_num, header= -1)
    A = pd.read_csv("PEMSD7/W_25.csv", header= -1)          #获取邻接矩阵
    

    A = np.array(A)
    A = torch.tensor(A, dtype=torch.float32)
       
    v = np.array(v)
    v = v.T
    v = torch.tensor(v, dtype=torch.float32)
    # 最终 v shape:[N, T]。  N=25, T=row_num
    
    
    # 模型参数
    A = A           # 邻接矩阵
    in_channels=1   # 输入通道数。只有速度信息，所以通道为1
    embed_size=64   # Transformer通道数
    time_num = 288  # 1天时间间隔数量
    num_layers=1    # Spatial-temporal block 堆叠层数
    T_dim=12        # 输入时间维度。 输入前1小时数据，所以 60min/5min = 12
    output_T_dim=3  # 输出时间维度。预测未来15,30,45min速度
    heads=1         # transformer head 数量。 时、空transformer头数量相同
    
    # model input shape: [1, N, T]   
    # 1:初始通道数, N:传感器数量, T:时间数量
    # model output shape: [N, T]    
    model = STTransformer(
        A,
        in_channels, 
        embed_size, 
        time_num, 
        num_layers, 
        T_dim, 
        output_T_dim, 
        heads
    )   
    
    # optimizer, lr, loss按论文要求
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()                             
    
    
    #   ----训练部分----
    # i表示遍历到的具体时间
    for t in range(train_num - 21):
        x = v[:, t:t+12]
        x = x.unsqueeze(0)        
        y = v[:, t+14:t+21:3]
        # x shape:[1, N, T_dim] 
        # y shape:[N, output_T_dim]
        
        out = model(x, t)
        loss = criterion(out, y) 
        
        if t%100 == 0:
            print("MAE loss:", loss)
        
        #常规操作
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
    
    
    
    #保存模型
    torch.save(model, "model.pth")
    
    
    
    
    
    
    
    
    
    