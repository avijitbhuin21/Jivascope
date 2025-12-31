"""
Attention Layer for LightCardiacNet

Implements temporal attention mechanism for Bi-GRU outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, rnn_output: torch.Tensor) -> tuple:
        weights = F.softmax(self.attention(rnn_output), dim=1)
        context = torch.sum(weights * rnn_output, dim=1)
        return context, weights.squeeze(-1)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size * 2, hidden_size)
        self.key = nn.Linear(hidden_size * 2, hidden_size)
        self.value = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size * 2)
    
    def forward(self, rnn_output: torch.Tensor) -> tuple:
        batch_size, seq_len, _ = rnn_output.shape
        
        Q = self.query(rnn_output)
        K = self.key(rnn_output)
        V = self.value(rnn_output)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        output = self.fc_out(attention_output)
        context = output.mean(dim=1)
        
        mean_weights = attention_weights.mean(dim=1).mean(dim=1)
        
        return context, mean_weights
