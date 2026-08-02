import torch
import torch.nn as nn
import itertools
from torch.utils.data import DataLoader

import LWE_Sample_Manager, dataset, model
from training_setup import make_optimizer_and_scheduler
from metrics import accuracy_within_tolerance

LWE = LWE_Sample_Manager.LWE(10, 10, 251, 2, 3, 10)
LWE.generate()

integer_base = 81
ds = dataset.LWEDataset(LWE, integer_base)
loader = DataLoader(ds, batch_size=4, shuffle=True)
data_iter = itertools.cycle(loader)
# src_batch, tgt_batch = next(iter(loader))

salsa = model.SalsaModel(ds.vocab_size,1024, 512, 32, 8, 2, 8)
optimizer, scheduler = make_optimizer_and_scheduler(salsa, 1e-5, 50)
criterion = nn.CrossEntropyLoss()

epoch_size = 40
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0.0
    total_accuracy = 0.0
    samples_seen = 0
    num_batchs = 0

    while samples_seen < epoch_size:
        src_batch, tgt_batch = next(data_iter)
        logits = salsa(src_batch, tgt_batch)

        logits_for_loss = logits[:, :-1, :]
        targets_for_loss = tgt_batch[:, 1:]

        loss = criterion(logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),targets_for_loss.reshape(-1),)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc = accuracy_within_tolerance(logits, tgt_batch, ds, tau=0.1)

        total_loss += loss.item()
        num_batchs += 1
        total_accuracy += acc
        samples_seen += src_batch.shape[0]

    avg_loss = total_loss / num_batchs
    avg_accuracy = total_accuracy / num_batchs
    print(f"epoch {epoch}: avg loss = {avg_loss:.4f}, "
          f"acc_tau=0.1 = {avg_accuracy:.2%} ({samples_seen} samples seen)")