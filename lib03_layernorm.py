import math
import torch
import torch_musa
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        # d_model: 特征维度
        # gamma: (d_model,) - 可学习的缩放参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        # beta: (d_model,) - 可学习的偏移参数
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        # 输入x维度: (batch_size, seq_len, d_model)
        # 或者其他形状，但最后一维必须是d_model
        
        # mean维度: (..., 1)
        # keepdim=True保持维度，在最后一维上计算均值
        # # LayerNorm公式:
        # y = gamma * (x - E[x])/sqrt(Var[x] + eps) + beta
        # 其中:
        # E[x]: 在特征维度上的均值
        # Var[x]: 在特征维度上的方差
        # gamma: 可学习的缩放参数
        # beta: 可学习的偏移参数
        # eps: 数值稳定性的小常数
        
        # 1. 计算均值 E[x]
        mean = x.mean(-1, keepdim=True)
        
        # 2. 计算方差 Var[x]
        var = x.var(-1, unbiased=False, keepdim=True)
        
        # 3. 标准化: (x - E[x])/sqrt(Var[x] + eps)
        out = (x - mean)/torch.sqrt(var+self.eps)
        
        # 4. 缩放和偏移: gamma * norm + beta
        out = self.gamma*out + self.beta
        return out