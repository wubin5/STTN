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
import matplotlib.pyplot as plt


def MAE(x, y):   #自己做MAE
    out = torch.abs(x-y)
    return out.mean(dim=0)  #求行平均值，输出每个预测时间MAE


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    days = 10
    val_days = 3    #需要验证天数
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
    
           
    #加载模型
    model = torch.load('model.pth')
    
    criterion1 = nn.L1Loss()    #MAE
    #criterion2 =               #MAPE 
    criterion3 = nn.MSELoss()   #RMSE
    
    pltx=[]
    plty=[]
    for t in range( train_num , row_num-21  ):
        x = v[:, t:t+12]
        x = x.unsqueeze(0)
        y = v[:, t+14:t+21:3]
        
        out = model(x, t)
                
        loss1 = criterion1(out, y ) 
        loss2 = MAE(out, y)
        loss3 = torch.sqrt(criterion3(out, y ) )
        
        if t%100 == 0:
            print("MAE loss", loss1)
            print("MAE loss2:", loss2)
            print("RMSE loss", loss3)
            print("\n")
        pltx.append(t)
        plty.append(loss1.detach().numpy())
        
        if t%288 == 0:         #画出每天MAE图
            plt.plot(pltx, plty, label=t)
            pltx.clear()
            plty.clear()
    
    #plt.plot(pltx, plty, label="STTN test")
    #plt.scatter(pltx, plty)

    plt.title("ST-Transformer test")
    plt.xlabel("t")
    plt.ylabel("MAE loss")
    plt.legend()
    plt.show()    
    
    #print("输出结果",out)
    #print("输出形状", out.shape)

    
    
    
    
    
    
    
    
    
    