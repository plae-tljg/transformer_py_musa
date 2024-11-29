import math
import torch
import torch_musa
from torch import nn
import torch.nn.functional as F
from lib02_multihead_attention import MultiHeadAttention
from lib03_layernorm import LayerNorm
from lib04_encoder import PositionwiseFeedForward
from lib01_transformer_embed import TransformerEmbedding

device = "musa"

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.dropout3 = nn.Dropout(drop_prob)
        self.norm3 = LayerNorm(d_model)
    
    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        x = self.norm1(x+_x)
        _x = x
        x = self.cross_attention(x, enc, enc, s_mask)
        x = self.dropout2(x)
        x = self.norm2(x+_x)
        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x+_x)
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device):
        super().__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(0, n_layer)])
        self.fc = nn.Linear(d_model, dec_voc_size)
        print("reached here to decoder")

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        x = dec
        for layer in self.layers:
            x = layer(x, enc, t_mask, s_mask)
        x = self.fc(x)
        return x
