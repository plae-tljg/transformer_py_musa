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
        # 输入维度:
        # dec: (batch_size, tgt_seq_len, d_model)
        # enc: (batch_size, src_seq_len, d_model)
        # t_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        # s_mask: (batch_size, 1, tgt_seq_len, src_seq_len)
        
        # 自注意力机制
        _x = dec
        # attention1输出: (batch_size, tgt_seq_len, d_model)
        x = self.attention1(dec, dec, dec, t_mask)
        x = self.dropout1(x)
        # norm1输出: (batch_size, tgt_seq_len, d_model)
        x = self.norm1(x + _x)
        
        # 交叉注意力机制
        _x = x
        # cross_attention输出: (batch_size, tgt_seq_len, d_model)
        x = self.cross_attention(x, enc, enc, s_mask)
        x = self.dropout2(x)
        # norm2输出: (batch_size, tgt_seq_len, d_model)
        x = self.norm2(x + _x)
        
        # 前馈网络
        _x = x
        # ffn输出: (batch_size, tgt_seq_len, d_model)
        x = self.ffn(x)
        x = self.dropout3(x)
        # norm3输出: (batch_size, tgt_seq_len, d_model)
        x = self.norm3(x + _x)
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device):
        super().__init__()
        self.embedding = TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(0, n_layer)])
        # 最后的线性层: d_model -> dec_voc_size
        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        # 输入维度:
        # dec: (batch_size, tgt_seq_len)
        # enc: (batch_size, src_seq_len, d_model)
        # t_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        # s_mask: (batch_size, 1, tgt_seq_len, src_seq_len)
        
        # embedding输出: (batch_size, tgt_seq_len, d_model)
        dec = self.embedding(dec)
        
        # 通过所有decoder层
        # 每层输出维度: (batch_size, tgt_seq_len, d_model)
        x = dec
        for layer in self.layers:
            x = layer(x, enc, t_mask, s_mask)
            
        # 最终线性层输出: (batch_size, tgt_seq_len, dec_voc_size)
        x = self.fc(x)
        return x
