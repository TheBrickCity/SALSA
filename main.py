import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import LWE_Sample_Manager
import dataset
import model
from training_setup import make_optimizer_and_scheduler
from recovery import direct_secret_recovery, verify_secret

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

n, m, q, h, sigma, gaussian_bound = 6, 200, 11, 1, 1, 3

lwe = LWE_Sample_Manager.LWE(n, m, q, h, sigma, gaussian_bound)
lwe.generate()
print("true secret:", lwe.s.tolist())

integer_base = 81
ds = dataset.LWEDataset(lwe, integer_base)
print("digits_per_int:", ds.digits_per_int, "| vocab_size:", ds.vocab_size)

batch_size = 16

loader = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device.type == "cuda"),
)

data_iter = itertools.cycle(loader)

salsa_model = model.SalsaModel(ds.vocab_size).to(device)

epoch_size = 1600
num_epochs = 50
warmup_steps = 200

optimizer, scheduler = make_optimizer_and_scheduler(
    salsa_model,
    1e-5,
    warmup_steps=warmup_steps,
)

criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    salsa_model.train()

    total_loss = 0.0
    samples_seen = 0

    while samples_seen < epoch_size:
        src_batch, tgt_batch = next(data_iter)

        src_batch = src_batch.to(device, non_blocking=True)
        tgt_batch = tgt_batch.to(device, non_blocking=True)

        logits = salsa_model(src_batch, tgt_batch)

        logits_for_loss = logits[:, :-1, :]
        targets_for_loss = tgt_batch[:, 1:]

        loss = criterion(
            logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),
            targets_for_loss.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        samples_seen += src_batch.shape[0]

    if epoch % 5 == 0 or epoch == num_epochs - 1:
        print(f"epoch {epoch}: avg loss = {total_loss / (samples_seen // batch_size):.4f}")

salsa_model.eval()

K_values = [2, 4, 6, 8, 10]

with torch.no_grad():
    guesses = direct_secret_recovery(salsa_model, ds, lwe.n, K_values)

print(f"\ngenerated {len(guesses)} candidate guesses from {len(K_values)} K values")

best_guess, best_std, recovered, all_scored = verify_secret(
    guesses, lwe.A, lwe.b, lwe.q, lwe.sigma
)

print("\nall guesses ranked by residual std (best first):")
for guess, std in all_scored:
    print(f"  {guess}  std={std:.3f}")

print("\nbest guess:      ", best_guess)
print("true secret:     ", lwe.s.tolist())
print("residual std:    %.3f" % best_std)
print("looks recovered: ", recovered)
print("exact match:     ", best_guess == lwe.s.tolist())