import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import norm_Adj
from models.DGCN import spatialAttentionScaledGCN, Spatial_Attention_layer, PositionWiseGCNFeedForward

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape
        x = x.reshape((-1, num_of_timesteps, in_channels))
        # (b, n, t, f)->(b*n,t,f_in)

        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)

        _, _, out_of_timesteps = x.shape
        x = x.transpose(1, 2).reshape(batch_size,  num_of_vertices, out_of_timesteps, in_channels)


        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, dgcn, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.dgcn = dgcn

    def forward(self, x, norm_adj, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)

        y = self.dgcn(y, norm_adj)

        # y = self.dropout(self.activation(self.conv1(y.transpose(-1,-2))))
        # y = self.dropout(self.conv2(y).transpose(-1,-2))

        # return self.norm2(x+y), attn
        return self.norm2(y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, edge_index, weights, attn_mask=None):
        # x [B, L, D]
        norm_adj = torch.from_numpy(norm_Adj(edge_index, weights, x)).type(torch.FloatTensor).to(x.device)
        attns = []  # 记录每层attention的结果
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, norm_adj, attn_mask=attn_mask)
                x = conv_layer(x)  # 进行Self-attention distilling来减小内存的占用 x:[batch_size, seq_len/2, d_model]
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, norm_adj, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, norm_adj, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
