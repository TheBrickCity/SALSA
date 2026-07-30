import LWE_Sample_Manager, dataset, time, model
from torch.utils.data import DataLoader

import model

# start = time.time()
LWE = LWE_Sample_Manager.LWE(10, 10, 251, 2, 3, 10)
# print(LWE.n,LWE.m, LWE.q,LWE.h, LWE.sigma)
LWE.generate()
# print("Secret", LWE.s)
# print("Matrix A", LWE.A)
# print("Error", LWE.e)
# print("Vector b", LWE.b)

integer_base = 81 # base that integer b and integers from vecors in A are converted into before tokenizing
ds = dataset.LWEDataset(LWE, integer_base)
loader = DataLoader(ds, batch_size=4, shuffle=True)
src_batch, tgt_batch = next(iter(loader))

print("src_batch shape:", src_batch.shape)
print("tgt_batch shape:", tgt_batch.shape)

# encoder side (1024 dim)
src_embedding = model.TokenEmbedding(ds.vocab_size, 1024)
encoder = model.SalsaEncoder(dimension=1024, num_heads=32, layer2_loops=2)

src_embedded = src_embedding(src_batch)
memory = encoder(src_embedded)
print("memory shape:", memory.shape)

# decoder side (512 dim)
tgt_embedding = model.TokenEmbedding(ds.vocab_size, 512)
decoder = model.SalsaDecoder(dimension=512, num_heads=8, layer2_loops=8)

tgt_embedded = tgt_embedding(tgt_batch)
print("tgt_embedded shape:", tgt_embedded.shape)

decoded = decoder(tgt_embedded, memory)
print("decoded shape:", decoded.shape)
print(decoded[0, 0][:5])
