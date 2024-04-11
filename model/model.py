import torch.nn as nn
from model.EncDec import Model
from model.embed import DataEmbedding

class Former(nn.Module):
    def __init__(self,d_model,d_ff,e_layers,num_fea,dropout=0):
        super(Former, self).__init__()
        self.embedding = DataEmbedding(num_fea,d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Model(d_model,d_ff,e_layers,num_fea)

    def forward(self,x):

        x_enc = self.embedding(x)
        Global, Local, Global_Attn_list, Local_Attn_list = self.encoder(x_enc)
        return Global, Local, Global_Attn_list, Local_Attn_list




















