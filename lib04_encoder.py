import math
import torch
import torch_musa
from torch import nn
import torch.nn.functional as F
from lib02_multihead_attention import MultiHeadAttention
from lib03_layernorm import LayerNorm
from lib01_transformer_embed import TransformerEmbedding

device = torch.device("musa")

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x,x,x,mask)
        x = self.dropout1(x)
        x = self.norm1(x+_x)
        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x+_x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, dropout, device):
        super().__init__()
        self.embedding = TransformerEmbedding(enc_voc_size, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, dropout) for _ in range(0, n_layer)])
    
    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
