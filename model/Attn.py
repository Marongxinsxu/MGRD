import math
from math import sqrt
import torch.nn as nn
import torch
import torch.nn.functional as F



class EncoderLayer(nn.Module):
    def __init__(self, attention, num_fea,  d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # Global
        self.G_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.G_conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.G_norm1 = nn.LayerNorm(d_model)
        self.G_norm2 = nn.LayerNorm(d_model)
        self.G_dropout = nn.Dropout(dropout)
        self.G_activation = F.relu if activation == "relu" else F.gelu
        # Local
        self.L_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.L_conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.L_norm1 = nn.LayerNorm(d_model)
        self.L_norm2 = nn.LayerNorm(d_model)
        self.L_dropout = nn.Dropout(dropout)
        self.L_activation = F.relu if activation == "relu" else F.gelu


    def forward(self, Global, Local):

        new_Global, new_Local,  Global_Attn, Local_Attn = self.attention(
            Global, Global, Global,
            Local
        )
        # Global
        Global = Global + self.G_dropout(new_Global)
        Global_y = Global = self.G_norm1(Global)
        Global_y = self.G_dropout(self.G_activation(self.G_conv1(Global_y.transpose(-1, 1))))
        Global_y = self.G_dropout(self.G_conv2(Global_y).transpose(-1, 1))
        # Local
        Local = Local + self.L_dropout(new_Local)
        Local_y = Local = self.L_norm1(Local)
        Local_y = self.L_dropout(self.L_activation(self.L_conv1(Local_y.transpose(-1, 1))))
        Local_y = self.L_dropout(self.L_conv2(Local_y).transpose(-1, 1))

        return self.G_norm2(Global + Global_y), self.L_norm2(Local + Local_y), Global_Attn, Local_Attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm1 = norm_layer
        self.norm2 = norm_layer

    def forward(self, x):
        Global_Attn_list = []
        Local_Attn_list = []

        Global = x
        Local = x
        for attn_layer in self.attn_layers:
            Global, Local, Global_Attn, Local_Attn = attn_layer(Global, Local)
            Global_Attn_list.append(Global_Attn)
            Local_Attn_list.append(Local_Attn)

        if self.norm1 is not None:
            Global = self.norm1(Global)
            Local = self.norm2(Local)

        return Global, Local, Global_Attn_list, Local_Attn_list

class AnomalyAttention(nn.Module):

    def __init__(self,  win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False ):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.distances = torch.zeros((win_size, win_size)).cuda()
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, G_queries, G_keys, G_values, L_values, sigma):
        B, L, H, E = G_queries.shape
        _, S, _, D = G_values.shape
        scale = self.scale or 1. / sqrt(E)

        G_scores = torch.einsum("blhe,bshe->bhls", G_queries, G_keys)

        G_attn = scale * G_scores

        sigma = sigma.transpose(1, 2)
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, self.window_size)
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        Local = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        Local = Local + G_attn
        Local_Attn = self.dropout(torch.softmax(Local, dim=-1))
        Global_Attn = self.dropout(torch.softmax(G_attn, dim=-1))

        # Global attn → V
        G_V = torch.einsum("bhls,bshd->blhd", Global_Attn, G_values)
        # (Global+Local)attn → V
        L_V = torch.einsum("bhls,bshd->blhd", Local_Attn, L_values)

        if self.output_attention:
            return (G_V.contiguous(), L_V.contiguous(), Global_Attn, Local_Attn)
        else:
            return (G_V.contiguous(), L_V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, num_fea, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        # Global
        self.G_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.G_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.G_value_projection = nn.Linear(d_model, d_values * n_heads)
        # Local

        self.L_value_projection = nn.Linear(d_model, d_values * n_heads)

        self.sigma_projection = nn.Linear(d_model, n_heads)

        self.G_out_projection = nn.Linear(d_values * n_heads, d_model)
        self.L_out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, G_queries, G_keys, G_values, L_values):
        B, L, _ = G_queries.shape
        _, S, _ = G_keys.shape
        H = self.n_heads

        x = G_queries
        ## σ
        sigma = self.sigma_projection(x).view(B, L, H)
        ## Global
        G_queries = self.G_query_projection(G_queries).view(B, L, H, -1)
        G_keys = self.G_key_projection(G_keys).view(B, S, H, -1)
        G_values = self.G_value_projection(G_values).view(B, S, H, -1)

        ## Local
        L_values = self.L_value_projection(L_values).view(B, S, H, -1)

        Global, Local, Global_Attn, Local_Attn = self.inner_attention(
            G_queries,
            G_keys,
            G_values,
            L_values,
            sigma
        )

        Global = Global.view(B, L, -1)
        Local = Local.view(B, L, -1)

        return self.G_out_projection(Global), self.L_out_projection(Local), Global_Attn, Local_Attn
