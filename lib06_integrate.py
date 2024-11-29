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
        
        # 创建k的mask
        k_mask = k != pad_idx  # [batch_size, k_len]
        
        # 扩展维度以匹配注意力分数的形状
        k_mask = k_mask.unsqueeze(1)  # [batch_size, 1, k_len]
        k_mask = k_mask.expand(-1, len_q, -1)  # [batch_size, q_len, k_len]
        
        return k_mask
    
    def make_casual_mask(self, q, k):
        """
        创建因果mask用于解码器的自注意力
        q: query序列 [batch_size, q_len]
        k: key序列 [batch_size, k_len]
        返回: mask [batch_size, q_len, k_len]
        """
        len_q, len_k = q.size(1), k.size(1)
        
        # 创建下三角矩阵并确保在正确的设备上
        mask = torch.triu(
            torch.ones(len_q, len_k, device=self.device), diagonal=1
        ).bool()
        
        # 扩展到batch维度
        mask = mask.unsqueeze(0)  # [1, q_len, k_len]
        mask = mask.expand(q.size(0), -1, -1)  # [batch_size, q_len, k_len]
        
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
        # 获取实际的序列长度
        src_len = src.size(1)  # 50
        trg_len = trg.size(1)  # 49
        
        # 创建适当维度的mask
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx)  # [batch_size, src_len, src_len]
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx)  # [batch_size, trg_len, trg_len]
        
        # Combine padding mask with causal mask for decoder
        casual_mask = self.make_casual_mask(trg, trg)  # [batch_size, trg_len, trg_len]
        trg_mask = trg_mask & casual_mask
        
        # 创建用于cross-attention的mask
        cross_mask = self.make_pad_mask(trg, src, self.src_pad_idx)  # [batch_size, trg_len, src_len]
        
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, cross_mask)
        return out
    
    @torch.no_grad()
    def generate(self, tokenizer, input_ids, max_length=50):
        # 初始化生成序列
        generated_ids = torch.tensor([[tokenizer.bos_token_id]], device=input_ids.device)
        
        # 获取编码器输出
        encoder_output = self.forward(input_ids, generated_ids)
        
        # 生成序列
        for _ in range(max_length):
            # 使用forward获取下一个token的预测
            decoder_output = self.forward(input_ids, generated_ids)
            next_token_logits = decoder_output[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # 添加预测的token到生成序列
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
            
            # 如果生成了结束符则停止
            if next_token_id.item() == tokenizer.eos_token_id:
                break

        return generated_ids