import LWE_Sample_Manager, dataset, time, model
from torch.utils.data import DataLoader

import model

start = time.time()
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
print("src shape:", src_batch.shape)
print(src_batch[0])

embedding = model.TokenEmbedding(ds.vocab_size, 1024)
embedded = embedding(src_batch)
print(embedded.shape)

encoder = model.BaselineEncoder(1024, 32)
encoded = encoder(embedded)
print(encoded.shape)
print(encoded[0,0][:5])
