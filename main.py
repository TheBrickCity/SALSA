import itertools
import torch.nn as nn
from torch.utils.data import DataLoader

import LWE_Sample_Manager, dataset, model
from training_setup import make_optimizer_and_scheduler
from recovery import direct_secret_recovery, verify_secret

n, m, q, h, sigma, gaussian_bound = 6, 200, 11, 1, 1, 3

lwe = LWE_Sample_Manager.LWE(n, m, q, h, sigma, gaussian_bound)
lwe.generate()
print("true secret:", lwe.s.tolist())

integer_base = 81
ds = dataset.LWEDataset(lwe, integer_base)
print("digits_per_int:", ds.digits_per_int, "| vocab_size:", ds.vocab_size)

batch_size = 16
loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
data_iter = itertools.cycle(loader)

salsa_model = model.SalsaModel(ds.vocab_size)

epoch_size = 1600
num_epochs = 50
warmup_steps = 200

optimizer, scheduler = make_optimizer_and_scheduler(salsa_model, 1e-5, warmup_steps=warmup_steps)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    total_loss = 0.0
    samples_seen = 0

    while samples_seen < epoch_size:
        src_batch, tgt_batch = next(data_iter)
        logits = salsa_model(src_batch, tgt_batch)

        logits_for_loss = logits[:, :-1, :]
        targets_for_loss = tgt_batch[:, 1:]
        loss = criterion(
            logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),
            targets_for_loss.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        samples_seen += src_batch.shape[0]

    if epoch % 5 == 0 or epoch == num_epochs - 1:
        print(f"epoch {epoch}: avg loss = {total_loss / (samples_seen // batch_size):.4f}")

K_values = [2, 4, 6, 8, 10]
guesses = direct_secret_recovery(salsa_model, ds, lwe.n, K_values)
print(f"\ngenerated {len(guesses)} candidate guesses from {len(K_values)} K values")

best_guess, best_std, recovered, all_scored = verify_secret(
    guesses, lwe.A, lwe.b, lwe.q, lwe.sigma
)

print("\nall guesses ranked by residual std (best first):")
for guess, std in all_scored:
    print(f"  {guess}  std={std:.3f}")

print("\nbest guess:      ", best_guess)
print("true secret:      ", lwe.s.tolist())
print("residual std:      %.3f" % best_std)
print("looks recovered:  ", recovered)
print("exact match:      ", best_guess == lwe.s.tolist())