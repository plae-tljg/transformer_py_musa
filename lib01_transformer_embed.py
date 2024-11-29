import torch
import torch_musa
from torch import nn, Tensor
import math

device = torch.device("musa")

class TokenEmbedding(nn.Embedding): # change index of a vocab list to an embedding
    def __init__(self, vocab_size, d_model):
        super().__init__(vocab_size, d_model, padding_idx=1)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):   # d_model for dim of model, max_length for length of sentence
        super().__init__()
        self.encoding = torch.zeros((max_len, d_model), device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)   # generate a sequence
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self, x):
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
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        # # ====== Debug Information ======
        # print("="*50)
        # print("[DEBUG] TransformerEmbedding dimensions:")
        # print(f"  Input x: {x.size()}")
        # print(f"  Token embedding: {tok_emb.size()}")
        # print(f"  Position embedding: {pos_emb.size()}")
        # print("="*50)
        return self.drop_out(tok_emb+pos_emb)
        


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

