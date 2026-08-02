import torch


def accuracy_within_tolerance(logits, tgt_batch, ds, tau=0.1):
    digit_logits = logits[:, :ds.digits_per_int, :] # grabs logits for real prediction, exlcuding SOS/EOS
    predicted_digit_tokens = digit_logits.argmax(dim=-1) # takes the highest logit for each position and makes it the predicted tokenID

    true_digit_tokens = tgt_batch[:, 1:1 + ds.digits_per_int] # grabs what the true tokenID's should be

    bound = tau * ds.q # tolerance for correctness
    batch_size = tgt_batch.shape[0]
    within_tolerance = 0

    for i in range(batch_size):
        predicted_digits = predicted_digit_tokens[i].tolist()  # tensor -> python list
        true_digits = true_digit_tokens[i].tolist() # tensor -> list

        predicted_b = ds._decode_int(predicted_digits) # tokenID/base B -> base 10
        true_b = ds._decode_int(true_digits) # tokenID/base B -> base 10

        if abs(predicted_b - true_b) <= bound: # check if within tolerance
            within_tolerance += 1

    return within_tolerance / batch_size