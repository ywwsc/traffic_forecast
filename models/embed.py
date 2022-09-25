import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F
import numpy as np

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :, :x.size(2)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, num_of_vertices):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = GCNConv(in_channels=d_model, out_channels=d_model)
        # self.linear = nn.Linear(3*c_in, d_model)
        self.embedding = torch.nn.Embedding(num_of_vertices, d_model)
        # padding = 1 if torch.__version__>='1.5.0' else 2
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                             kernel_size=3, padding=padding, padding_mode='circular')
        # for m in self.modules():
        #     if isinstance(m, GCNConv):
        #         nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x, edge_index, weights):

        batch, num_of_vertices, timestamps, _ = x.shape
        x_indexs = torch.LongTensor(torch.arange(num_of_vertices)).to(x.device)  # (N,)
        embed = self.embedding(x_indexs).unsqueeze(0)  # (N, d_model)->(1,N,d_model)
        embed = self.tokenConv(embed, edge_index, weights)
        embed = embed.unsqueeze(2)

        # y = torch.unsqueeze(y, 3)
        # y = F.relu(y)
        # y = y.reshape(y.shape[0], y.shape[1], y.shape[2]*y.shape[3])
        # y = self.linear(y)
        # x = F.softmax(x, dim=-1)
        return embed
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

# class PoiEmbedding(nn.Module):
#     def __init__(self, d_model):
#         super(PoiEmbedding, self).__init__()
#         self.embed = nn.Linear(24, d_model)
#     def forward(self):
#
#         return self.embed(poi_data)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, poi=False, node_num=0):
        super(DataEmbedding, self).__init__()

        self.linear = nn.Linear(c_in, d_model)
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, num_of_vertices=node_num)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        # if poi is not None:
        #     self.poi_embedding = PoiEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, edge_index, weights):
        x = self.linear(x)
        x = x + self.position_embedding(x) + self.value_embedding(x, edge_index, weights) \
            # + self.temporal_embedding(x_mark) \
            # + self.poi_embedding() if self.poi_embedding is not False  \
            # else self.value_embedding(x, edge_index, weights) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x.detach())

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)
