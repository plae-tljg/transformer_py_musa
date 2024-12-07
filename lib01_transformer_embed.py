import torch
import torch_musa
from torch import nn, Tensor
import math

device = torch.device("musa")

class TokenEmbedding(nn.Embedding): # change index of a vocab list to an embedding
    def __init__(self, vocab_size, d_model):
        # vocab_size: 词汇表大小
        # d_model: 嵌入维度
        # 输出维度: (batch_size, seq_len, d_model)
        super().__init__(vocab_size, d_model, padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):   # d_model for dim of model, max_length for length of sentence
        super().__init__()
        # encoding维度: (max_len, d_model)
        self.encoding = torch.zeros((max_len, d_model), device=device)
        self.encoding.requires_grad = False
        
        # pos维度: (max_len, 1)
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        
        # _2i维度: (d_model/2,)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        
        # 最终encoding维度保持: (max_len, d_model)
        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self, x):
        # 输入x维度: (batch_size, seq_len)
        # 输出维度: (seq_len, d_model)
        batch_size, seq_len = x.size()
        # # ====== Debug Information ======
        # print("="*50)
        # print("[DEBUG] PositionalEmbedding dimensions:")
        # print(f"  Input x: {x.size()}")
        # print(f"  Encoding shape: {self.encoding.size()}")
        # print(f"  Sequence length: {seq_len}")
        # print("="*50)
        
        # 确保编码维度与输入序列长度匹配
        if seq_len > self.encoding.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.encoding.size(0)}")
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        # 输入x维度: (batch_size, seq_len)
        
        # tok_emb维度: (batch_size, seq_len, d_model)
        tok_emb = self.tok_emb(x)
        
        # pos_emb维度: (seq_len, d_model)
        pos_emb = self.pos_emb(x)
        
        # pos_emb会自动广播到tok_emb的维度
        # 最终输出维度: (batch_size, seq_len, d_model)
        return self.drop_out(tok_emb + pos_emb)
        


if __name__ == '__main__':
    #region test 1, generate torch tensors/matrices
    # random_torch = torch.rand(4, 4) # generate 4by4 matrix
    # print(random_torch)
    #endregion

    #region test 2, test TokenEmbedding
    # layer = TokenEmbedding(10, 19)
    # input_indices = torch.LongTensor([[1, 5, 2, 0, 1], [6, 2, 1, 8, 9]]) # Batch size 2, sequence length 5
    # embeddings = layer(input_indices)
    # print(embeddings.shape)  # Output: torch.Size([2, 5, 300])
    # print(embeddings)
    #endregion

    print("test finished")

