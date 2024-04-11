import torch.nn as nn
from model.Attn import Encoder,EncoderLayer,AttentionLayer,AnomalyAttention
import torch


class Model(nn.Module):
    def __init__(self,d_model,d_ff,e_layers,num_fea,win_size=100,dropout=0,activation='gelu', output_attention=True,n_heads=8):
        super(Model, self).__init__()

        self.output_attention = output_attention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout,
                                                 output_attention=output_attention),num_fea,
                        d_model, n_heads),
                    num_fea,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection1 = nn.Linear(d_model, num_fea, bias=True)
        self.projection2 = nn.Linear(d_model, num_fea, bias=True)


    def forward(self,x):

        Global, Local, Global_Attn, Local_Attn = self.encoder(x)
        Global = self.projection1(Global)
        Local = self.projection2(Local)

        if self.output_attention:
            return Global, Local, Global_Attn, Local_Attn
        else:
            return Global, Local



























