import math
import torch
import torch_musa
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine =nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        # # ====== Debug Information ======
        # print("="*50)
        # print("[DEBUG] Input dimensions:")
        # print("="*50) 
        # print(f"  q: {q.size()}")
        # print("-"*50)
        # print(f"  k: {k.size()}")
        # print("-"*50) 
        # print(f"  v: {v.size()}")
        # print("-"*50)
        # print(f"  batch_size: {batch}")
        # print("="*50)
        n_d = self.d_model//self.n_head
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, -1, self.n_head, n_d).permute(0,2,1,3)
        k = k.view(batch, -1, self.n_head, n_d).permute(0,2,1,3)
        v = v.view(batch, -1, self.n_head, n_d).permute(0,2,1,3)
        score = q@k.transpose(2,3)/math.sqrt(n_d)
        if mask is not None:
            mask = mask.unsqueeze(1)
            # print("mask dimension", mask.size())
            # print("score dimension", score.size())
            score = score.masked_fill(mask==0, -1e9)
        score = self.softmax(score)@v
        
        # # ====== Debug Information ======
        # print("="*50)
        # print("[DEBUG] MultiHeadAttention shapes:")
        # print("="*50)
        # print(f"  q: {q.size()}")
        # print("-"*50)
        # print(f"  k: {k.size()}")
        # print("-"*50)
        # print(f"  v: {v.size()}")
        # print("-"*50)
        # print(f"  score: {score.size()}")
        # print("-"*50)
        # print(f"  d_model: {self.d_model}, n_head: {self.n_head}")
        # print("-"*50)
        # print(f"  n_d (d_model/n_head): {n_d}")
        # print("="*50)
        
        score = score.permute(0,2,1,3).contiguous().view(batch, -1, self.d_model)
        # print(f"  final score: {score.shape}")
        # print("="*50)
        out = self.w_combine(score)
        return out

if __name__ == '__main__':
    d_model = 10
    n_head = 2
    seq_len = 10
    batch_size = 4
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    attention = MultiHeadAttention(d_model, n_head)
    out = attention(q, k, v)
    print(out)