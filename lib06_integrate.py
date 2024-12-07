import math
import torch
import torch_musa
from torch import nn
import torch.nn.functional as F
from lib02_multihead_attention import MultiHeadAttention
from lib03_layernorm import LayerNorm
from lib04_encoder import PositionwiseFeedForward, Encoder
from lib05_decoder import Decoder
from lib00_tokenize import Llama2Tokenizer


device = "musa"

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, d_model, max_len, n_head, ffn_hidden, n_layer, drop_prob, device):
        super().__init__()
        # enc_voc_size: 源语言词汇表大小
        # dec_voc_size: 目标语言词汇表大小
        self.encoder = Encoder(enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx):
        """
        创建padding mask以忽略padding token
        q: query序列 [batch_size, q_len]
        k: key序列 [batch_size, k_len]
        pad_idx: padding token id
        返回: mask [batch_size, q_len, k_len]
        """
        len_q, len_k = q.size(1), k.size(1)
        
        # k_mask初始维度: (batch_size, k_len)
        k_mask = k != pad_idx
        
        # 扩展维度:
        # 1. unsqueeze: (batch_size, 1, k_len)
        # 2. expand: (batch_size, q_len, k_len)
        k_mask = k_mask.unsqueeze(1).expand(-1, len_q, -1)
        return k_mask
    
    def make_casual_mask(self, q, k):
        """
        创建因果mask用于解码器的自注意力
        q: query序列 [batch_size, q_len]
        k: key序列 [batch_size, k_len]
        返回: mask [batch_size, q_len, k_len]
        """
        len_q, len_k = q.size(1), k.size(1)
        
        # 创建下三角矩阵: (q_len, k_len)
        # 扩展到batch维度:
        # 1. unsqueeze: (1, q_len, k_len)
        # 2. expand: (batch_size, q_len, k_len)
        mask = torch.triu(torch.ones(len_q, len_k, device=self.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).expand(q.size(0), -1, -1)
        return ~mask

    def make_trg_mask(self, trg):
        """
        生成目标序列的掩码矩阵
        Args:
            trg: 目标序列 [batch_size, trg_len]
        Returns:
            trg_mask: [batch_size, 1, trg_len, trg_len]
        """
        batch_size, trg_len = trg.size()
        # 创建下三角矩阵掩码
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        # 输入维度:
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        
        # mask维度:
        # src_mask: (batch_size, src_len, src_len)
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx)
        
        # trg_mask: (batch_size, trg_len, trg_len)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx)
        
        # casual_mask: (batch_size, trg_len, trg_len)
        casual_mask = self.make_casual_mask(trg, trg)
        
        # 组合mask: (batch_size, trg_len, trg_len)
        trg_mask = trg_mask & casual_mask
        
        # cross_mask: (batch_size, trg_len, src_len)
        cross_mask = self.make_pad_mask(trg, src, self.src_pad_idx)
        
        # 编码器输出维度: (batch_size, src_len, d_model)
        enc = self.encoder(src, src_mask)
        
        # 解码器输出维度: (batch_size, trg_len, dec_voc_size)
        out = self.decoder(trg, enc, trg_mask, cross_mask)
        return out
    
    @torch.no_grad()
    def generate(self, tokenizer, input_ids, max_length=50):
        # input_ids维度: (batch_size, src_len)
        # generated_ids初始维度: (1, 1)
        generated_ids = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
        
        # encoder_output维度: (batch_size, src_len, d_model)
        encoder_output = self.forward(input_ids, generated_ids)
        
        for _ in range(max_length):
            # decoder_output维度: (batch_size, curr_len, dec_voc_size)
            decoder_output = self.forward(input_ids, generated_ids)
            
            # next_token_logits维度: (batch_size, dec_voc_size)
            next_token_logits = decoder_output[:, -1, :]
            
            # next_token_id维度: (batch_size, 1)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # generated_ids维度: (batch_size, curr_len + 1)
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
            
            # 如果生成了结束符则停止
            if next_token_id.item() == tokenizer.eos_token_id:
                break

        return generated_ids