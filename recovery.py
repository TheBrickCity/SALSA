import torch
import numpy as np

def build_probe_src(n, ds, K, index):
    row = [0] * n
    row[index] = K

    src = []
    for j, val in enumerate(row):
        src.extend(ds._encode_int(int(val)))
        if j < n - 1:
            src.append(ds.SEP)

    return src

def direct_secret_recovery(salsa_model, ds, n, K_values):
    guesses = []

    for K in K_values:
        # build all n probe inputs for this K as one batch
        probe_srcs = [build_probe_src(n, ds, K, i) for i in range(n)]
        src_batch = torch.tensor(probe_srcs, dtype=torch.long)

        predicted_tokens = salsa_model.generate(
            src_batch, sos_token=ds.SOS, num_digits=ds.digits_per_int, base=ds.base
        )

        p = [ds._decode_int(predicted_tokens[i].tolist()) for i in range(n)]

        guesses.extend(binarize_mean(p))

    return guesses

def binarize_mean(p):
    mean_val = sum(p) / len(p)

    f01 = [0 if val > mean_val else 1 for val in p]  # above mean -> 0
    f10 = [1 if val > mean_val else 0 for val in p]  # above mean -> 1

    return [f01, f10]

def residual_std(s_guess, A, b, q):
    s = np.array(s_guess)
    r = (A @ s - b) % q
    r_signed = ((r + q // 2) % q) - q // 2

    return float(np.std(r_signed))

def verify_secret(guesses, A, b, q, sigma):
    scored = [(guess, residual_std(guess, A, b, q)) for guess in guesses]
    scored.sort(key=lambda pair: pair[1])

    best_guess, best_std = scored[0]
    uniform_std = q / np.sqrt(12)
    midpoint = (sigma + uniform_std) / 2
    recovered = best_std < midpoint

    return best_guess, best_std, recovered, scored