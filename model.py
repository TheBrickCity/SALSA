import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, dimension, max_len=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dimension)
        self.position_embed = nn.Embedding(max_len, dimension)
        self.dimension = dimension

    def forward(self, x):
        batch, seq_length = x.shape
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        positions = positions.expand(batch, seq_length)
        return self.token_embed(x) + self.position_embed(positions) # output size: batch x seq_length x dimension