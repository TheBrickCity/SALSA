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

# class BaselineEncoder(nn.Module):
#     def __init__(self, dimension, num_heads):
#         super().__init__()
#         layer = nn.TransformerEncoderLayer(d_model=dimension, nhead=num_heads, batch_first=True)
#         self.encoder = nn.TransformerEncoder(layer, num_layers=1)
#
#     def forward(self, x):
#         return self.encoder(x)

class SalsaEncoder(nn.Module):
    def __init__(self, dimension, num_heads, layer2_loops):
        super().__init__()
        self.layer1 = nn.TransformerEncoderLayer(d_model=dimension, nhead=num_heads, batch_first=True)
        self.layer2 = nn.TransformerEncoderLayer(d_model=dimension, nhead=num_heads, batch_first=True)
        self.layer2_loops = layer2_loops
        self.gate = nn.Linear(dimension, dimension)


    def forward(self, x):
        x = self.layer1(x)
        for _ in range(self.layer2_loops):
            new_x = self.layer2(x)
            g = torch.sigmoid(self.gate(x))
            x = g * new_x + (1-g) * x
        return x

class SalsaDecoder(nn.Module):
    def __init__(self, dimension, num_heads, layer2_loops):
        super().__init__()
        self.layer1 = nn.TransformerDecoderLayer(d_model=dimension, nhead=num_heads, batch_first=True)
        self.layer2 = nn.TransformerDecoderLayer(d_model=dimension, nhead=num_heads, batch_first=True)
        self.layer2_loops = layer2_loops

    def forward(self,tgt, memory):
        seq_len = tgt.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        x = self.layer1(tgt, memory, tgt_mask=mask)
        for _ in range(self.layer2_loops):
            x = self.layer2(x, memory, tgt_mask=mask)
        return x