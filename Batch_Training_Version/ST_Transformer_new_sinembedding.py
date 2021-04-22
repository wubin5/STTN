# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:28:06 2020

@author: wb
"""

import torch
import torch.nn as nn
from GCN_models import GCN
from One_hot_encoder import One_hot_encoder
import torch.nn.functional as F
import numpy as np

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        K: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        V: [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]
        attn_mask: [batch_size, n_heads, seq_len, seq_len] 可能没有
        '''
        B, n_heads, len1, len2, d_k = Q.shape 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        # scores : [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), N(Spatial) or T(Temporal)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, T(Spatial) or N(Temporal), N(Spatial) or T(Temporal), d_k]]
        return context



class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        # 用Linear来做投影矩阵    
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, T, N, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # Q: [B, h, T, N, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # K: [B, h, T, N, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)  # V: [B, h, T, N, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) # [B, h, T, N, d_k]
        context = context.permute(0, 3, 2, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
            
        # 用Linear来做投影矩阵    
        # 但这里如果是多头的话，是不是需要声明多个矩阵？？？

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4) # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention()(Q, K, V) #[B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4) #[B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim) # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context) # [batch_size, len_q, d_model]
        return output 



class STransformer(nn.Module):
    def __init__(self, embed_size, heads, adj, cheb_K, dropout, forward_expansion):
        super(STransformer, self).__init__()
        # Spatial Embedding
        self.adj = adj
        self.D_S = adj.to('cuda:0')
#         self.embed_liner = nn.Linear(adj.shape[0], embed_size)
        
        self.attention = SMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        # 调用GCN
        self.gcn = GCN(embed_size, embed_size*2, embed_size, adj, cheb_K, dropout)  
        self.norm_adj = nn.InstanceNorm2d(1)    # 对邻接矩阵归一化

        self.dropout = nn.Dropout(dropout)
        self.fs = nn.Linear(embed_size, embed_size)
        self.fg = nn.Linear(embed_size, embed_size)

    def forward(self, value, key, query):
        # value, key, query: [N, T, C]  [B, N, T, C]        
        # Spatial Embedding 部分 
#         N, T, C = query.shape
#         D_S = self.embed_liner(self.D_S) # [N, C]
#         D_S = D_S.expand(T, N, C) #[T, N, C]相当于在第一维复制了T份
#         D_S = D_S.permute(1, 0, 2) #[N, T, C]
        B, N, T, C = query.shape
#         D_S = self.embed_liner(self.D_S) # [N, C]
        D_S = get_sinusoid_encoding_table(N, C).to('cuda:0')
        D_S = D_S.expand(B, T, N, C) #[B, T, N, C]相当于在第2维复制了T份, 第一维复制B份
        D_S = D_S.permute(0, 2, 1, 3) #[B, N, T, C]
        
        
        # GCN 部分


        X_G = torch.Tensor(B, N,  0, C).to('cuda:0')
        self.adj = self.adj.unsqueeze(0).unsqueeze(0)
        self.adj = self.norm_adj(self.adj)
        self.adj = self.adj.squeeze(0).squeeze(0)
        
        for t in range(query.shape[2]):
            o = self.gcn(query[ : ,:,  t,  : ],  self.adj) # [B, N, C]
            o = o.unsqueeze(2)              # shape [N, 1, C] [B, N, 1, C]
#             print(o.shape)
            X_G = torch.cat((X_G, o), dim=2)
         # 最后X_G [B, N, T, C]   
        
#         print('After GCN:')
#         print(X_G)
        # Spatial Transformer 部分
        query = query + D_S
        attention = self.attention(query, query, query) #(B, N, T, C)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        U_S = self.dropout(self.norm2(forward + x))

        
        # 融合 STransformer and GCN  
        g = torch.sigmoid(self.fs(U_S) +  self.fg(X_G))      # (7)
        out = g*U_S + (1-g)*X_G                                # (8)

        return out #(B, N, T, C)    


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, dropout, forward_expansion):
        super(TTransformer, self).__init__()
        
        # Temporal embedding One hot
        self.time_num = time_num
#         self.one_hot = One_hot_encoder(embed_size, time_num)          # temporal embedding选用one-hot方式 或者
#         self.temporal_embedding = nn.Embedding(time_num, embed_size)  # temporal embedding选用nn.Embedding


        
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, t):
        B, N, T, C = query.shape
        
#         D_T = self.one_hot(t, N, T)                          # temporal embedding选用one-hot方式 或者
#         D_T = self.temporal_embedding(torch.arange(0, T).to('cuda:0'))    # temporal embedding选用nn.Embedding
        D_T = get_sinusoid_encoding_table(T, C).to('cuda:0')
        D_T = D_T.expand(B, N, T, C)


        # temporal embedding加到query。 原论文采用concatenated
        query = query + D_T  
        
        attention = self.attention(query, query, query)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out




### STBlock

class STTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, adj, time_num, cheb_K, dropout, forward_expansion):
        super(STTransformerBlock, self).__init__()
        self.STransformer = STransformer(embed_size, heads, adj, cheb_K, dropout, forward_expansion)
        self.TTransformer = TTransformer(embed_size, heads, time_num, dropout, forward_expansion)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, t):
    # value,  key, query: [N, T, C] [B, N, T, C]
        # Add skip connection,run through normalization and finally dropout
        x1 = self.norm1(self.STransformer(value, key, query) + query) #(B, N, T, C)
        x2 = self.dropout( self.norm2(self.TTransformer(x1, x1, x1, t) + x1) ) 
        return x2




### Encoder
class Encoder(nn.Module):
    # 堆叠多层 ST-Transformer Block
    def __init__(
        self,
        embed_size,
        num_layers,
        heads,
        adj,
        time_num,
        device,
        forward_expansion,
        cheb_K,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.layers = nn.ModuleList(
            [
                STTransformerBlock(
                    embed_size,
                    heads,
                    adj,
                    time_num,
                    cheb_K,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
    # x: [N, T, C]  [B, N, T, C]
        out = self.dropout(x)        
        # In the Encoder the query, key, value are all the same.
        for layer in self.layers:
            out = layer(out, out, out, t)
        return out     
    


### Transformer   
class Transformer(nn.Module):
    def __init__(
        self,
        adj,
        embed_size,
        num_layers,
        heads,
        time_num,
        forward_expansion, ##？
        cheb_K,
        dropout,
        
        device="cuda:0"
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            embed_size,
            num_layers,
            heads,
            adj,
            time_num,
            device,
            forward_expansion,
            cheb_K,
            dropout
        )
        self.device = device

    def forward(self, src, t): 
        ## scr: [N, T, C]   [B, N, T, C]
        enc_src = self.encoder(src, t) 
        return enc_src # [B, N, T, C]


### ST Transformer: Total Model

class STTransformer_sinembedding(nn.Module):
    def __init__(
        self, 
        adj,
        in_channels, 
        embed_size, 
        time_num,
        num_layers,
        T_dim,
        output_T_dim,  
        heads,    
        cheb_K,
        forward_expansion, 
        dropout = 0 
    ):        
        super(STTransformer_sinembedding, self).__init__()

        self.forward_expansion = forward_expansion
        # 第一次卷积扩充通道数
        self.conv1 = nn.Conv2d(in_channels, embed_size, 1)
        self.Transformer = Transformer(
            adj,
            embed_size, 
            num_layers, 
            heads, 
            time_num,
            forward_expansion,
            cheb_K,
            dropout = 0
        )

        # 缩小时间维度。  例：T_dim=12到output_T_dim=3，输入12维降到输出3维
        self.conv2 = nn.Conv2d(T_dim, output_T_dim, 1)  
        # 缩小通道数，降到1维。
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # input x shape[ C, N, T] 
        # C:通道数量。  N:传感器数量。  T:时间数量
        
#         x = x.unsqueeze(0)
        
        input_Transformer = self.conv1(x)        
#         input_Transformer = input_Transformer.squeeze(0)
#         input_Transformer = input_Transformer.permute(1, 2, 0)
        input_Transformer = input_Transformer.permute(0, 2, 3, 1)

        
        
        
        #input_Transformer shape[N, T, C]   [B, N, T, C]
        output_Transformer = self.Transformer(input_Transformer, self.forward_expansion)  # [B, N, T, C]
        output_Transformer = output_Transformer.permute(0, 2, 1, 3)
        #output_Transformer shape[B, T, N, C]
        
#         output_Transformer = output_Transformer.unsqueeze(0)     
        out = self.relu(self.conv2(output_Transformer))    # 等号左边 out shape: [1, output_T_dim, N, C]        
        out = out.permute(0, 3, 2, 1)           # 等号左边 out shape: [B, C, N, output_T_dim]
        out = self.conv3(out)                   # 等号左边 out shape: [B, 1, N, output_T_dim]   
        out = out.squeeze(1)
        
       
        return out #[B, N, output_dim]
        # return out shape: [N, output_dim]



