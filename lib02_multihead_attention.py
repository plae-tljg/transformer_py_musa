import math
import torch
import torch_musa
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        # d_model: 模型的维度
        # n_head: 注意力头的数量
        # 每个头的维度 n_d = d_model/n_head
        self.n_head = n_head
        self.d_model = d_model
        assert d_model % n_head == 0
        # 所有Linear层的维度转换：d_model -> d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q, k, v, mask=None):
        # 输入维度:
        # q: (batch, seq_len_q, d_model)
        # k: (batch, seq_len_k, d_model)
        # v: (batch, seq_len_v, d_model)
        # 注：通常 seq_len_k = seq_len_v
        batch = q.size(0)
        n_d = self.d_model//self.n_head  # 每个头的维度

        # 线性变换后维度仍然相同
        # q,k,v: (batch, seq_len, d_model)
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 重塑和转置操作:
        # 1. view: (batch, seq_len, n_head, n_d)
        # 2. permute: (batch, n_head, seq_len, n_d)
        q = q.view(batch, -1, self.n_head, n_d).permute(0,2,1,3)
        k = k.view(batch, -1, self.n_head, n_d).permute(0,2,1,3)
        v = v.view(batch, -1, self.n_head, n_d).permute(0,2,1,3)

        # 注意力计算:
        # 1. q@k.transpose: (batch, n_head, seq_len_q, seq_len_k)
        # 2. score经过softmax后: (batch, n_head, seq_len_q, seq_len_k)
        # 3. score@v: (batch, n_head, seq_len_q, n_d)
        score = q@k.transpose(2,3)/math.sqrt(n_d)
        
        if mask is not None:
            # mask: (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask==0, -1e9)
        score = self.softmax(score)@v
        
        # 重塑回
        score = score.permute(0,2,1,3).contiguous().view(batch, -1, self.d_model)
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