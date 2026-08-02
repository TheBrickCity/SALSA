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
        self.gate = nn.Linear(dimension, dimension)

    def forward(self,tgt, memory):
        seq_len = tgt.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tgt.device)
        x = self.layer1(tgt, memory, tgt_mask=mask)
        for _ in range(self.layer2_loops):
            new_x = self.layer2(x, memory, tgt_mask=mask)
            g = torch.sigmoid(self.gate(x))
            x = g * new_x + (1 - g) * x
        return x

class EncoderDecoderProjection(nn.Module): # temp fix to encoder/decoder dim mismatch
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)

    def forward(self,memory):
        return self.proj(memory)

class OutputLayer(nn.Module):
    def __init__(self, dimension, vocab_size):
        super().__init__()
        self.fc = nn.Linear(dimension, vocab_size)

    def forward(self, x):
        return self.fc(x)

class SalsaModel(nn.Module):
    def __init__(self, vocab_size, encoder_dim=1024, decoder_dim=512,encoder_heads=32, decoder_heads=8,encoder_loops=2, decoder_loops=8):
        super().__init__()
        self.src_embedding = TokenEmbedding(vocab_size, encoder_dim)
        self.encoder = SalsaEncoder(encoder_dim, encoder_heads, encoder_loops)
        self.projection = EncoderDecoderProjection(encoder_dim, decoder_dim)
        self.tgt_embedding = TokenEmbedding(vocab_size, decoder_dim)
        self.decoder = SalsaDecoder(decoder_dim, decoder_heads, decoder_loops)
        self.output_layer = OutputLayer(decoder_dim, vocab_size)

    def forward(self, src, tgt):
        src_embedded = self.src_embedding(src)
        memory = self.projection(self.encoder(src_embedded))
        tgt_embedded = self.tgt_embedding(tgt)
        decoded = self.decoder(tgt_embedded, memory)
        return self.output_layer(decoded)

    @torch.no_grad
    def generate(self, src, sos_token, num_digits, base):
        self.eval()
        batch_size = src.shape[0]
        src_embedded = self.src_embedding(src)
        memory = self.projection(self.encoder(src_embedded))
        generated = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=src.device)

        for _ in range(num_digits):
            tgt_embedded = self.tgt_embedding(generated)
            decoded = self.decoder(tgt_embedded, memory)
            logits = self.output_layer(decoded)

            next_token_logits = logits[:, -1, :base]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)
        self.train()
        return generated[:, 1:]