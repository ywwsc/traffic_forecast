import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class ExternalAttention(nn.Module):
    def __init__(self, d_model, n_heads ,d_keys=None, d_values=None, scale=None, attention_dropout=0.1, output_attention=False, mix=False):
        super(ExternalAttention, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.query_projection = nn.Linear(24, 1)
        self.query_projection2 = nn.Linear(1683, d_model)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads
        self.scale = scale
        self.mix = mix

    def forward(self, x, poi_data):
        B, L, _ = x.shape
        H = self.n_heads
        key = self.key_projection(x).view(B, L, H, -1)
        value = self.value_projection(x).view(B, L, H, -1)
        query = self.query_projection(poi_data)
        query = query.squeeze()
        query = self.query_projection2(query)
        query = torch.repeat_interleave(query.unsqueeze(dim=0), repeats=L, dim=0)  # 扩展并复制维数
        query = torch.repeat_interleave(query.unsqueeze(dim=0), repeats=B, dim=0).contiguous().view(B, L, H, -1)

        b, l, h, d = query.shape
        scale = self.scale or 1. / sqrt(d)

        scores = torch.einsum("blhe,bshe->bhls", query, key)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, value).contiguous()

        out = V.view(B, L, -1)

        # if self.output_attention:
        #     return (V.contiguous(), A)
        # else:
        #     return (V.contiguous(), None)

        return self.out_projection(out)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  # batch_size, seq_len, head_num, dim_feature
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn

class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[-3]
        channel = values.shape[-2]
        length = values.shape[-1]
        node_num = values.shape[1]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=-2), dim=-2)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]

        weights = torch.stack([torch.stack([mean_value[:, j, index[j, i]] for i in range(top_k)], dim=-1) for j in range(node_num)], dim=-2)
            #     weights = torch.stack(mean_value[:, j, index[i]]
            # weights = torch.stack([torch.stack([mean_value[:, j, index[i]] for i in range(top_k)], dim=-1)], dim=-2)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):
            pattern = []
            for j in range(node_num):
                pattern.append(torch.roll(tmp_values[:, j], -int(index[j, i]), -1))
            pattern = torch.stack(pattern, dim=1)
            delays_agg = delays_agg + pattern * \
                        (tmp_corr[:, :, i].unsqueeze(2).unsqueeze(2).unsqueeze(2).repeat(1, 1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[-3]
        channel = values.shape[-2]
        length = values.shape[-1]
        node_num = values.shape[1]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, node_num, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=-2), dim=-2)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, node_num, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, node_num, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        node_num = values.shape[1]
        head = values.shape[2]
        channel = values.shape[3]
        length = values.shape[4]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, node_num, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, N, L, H, E = queries.shape
        _, N, S, _, D = values.shape
        # starttime = time.perf_counter()
        if L > S:
            zeros = torch.zeros_like(queries[:, :, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=2)
            keys = torch.cat([keys, zeros], dim=2)
        else:
            values = values[:, :, :L, :, :]
            keys = keys[:, :, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 1, 3, 4, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 1, 3, 4, 2).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # print('start attn agg', time.time())
        # time delay agg
        if self.training:
            V = self.time_delay_agg_full(values.permute(0, 1, 3, 4, 2).contiguous(), corr).permute(0, 1, 4, 2, 3)
        else:
            V = self.time_delay_agg_full(values.permute(0, 1, 3, 4, 2).contiguous(), corr).permute(0, 1, 4, 2, 3)
        # endtime = time.perf_counter()
        # print('attn time', endtime - starttime)
        return V.contiguous(), corr.permute(0, 1, 4, 2, 3)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, N, L, _ = queries.shape
        _, N, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, N, L, H, -1)
        keys = self.key_projection(keys).view(B, N, S, H, -1)
        values = self.value_projection(values).view(B, N, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, N, L, -1)  # out的维度应该是[batch_size, seq_len, d_values*n_heads]

        return self.out_projection(out), attn
