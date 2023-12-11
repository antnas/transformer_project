import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_len):
        super().__init__()

        pe = torch.zeros(seq_len, embedding_dim)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        div_terms = 1 / torch.pow(10000, torch.arange(0, embedding_dim, 2) / embedding_dim)

        pe[:, 0::2] = torch.sin(pos * div_terms)
        pe[:, 1::2] = torch.cos(pos * div_terms)

        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]