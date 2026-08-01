import torch
import torch.nn as nn
import LWE_Sample_Manager, dataset, model
from torch.utils.data import DataLoader

lwe = LWE_Sample_Manager.LWE(10, 10, 251, 2, 3, 10)
lwe.generate()

integer_base = 81
ds = dataset.LWEDataset(lwe, integer_base)
loader = DataLoader(ds, batch_size=4, shuffle=True)
src_batch, tgt_batch = next(iter(loader))

salsa = model.SalsaModel(ds.vocab_size,1024, 512, 32, 8, 2, 8)
print(salsa(src_batch, tgt_batch))