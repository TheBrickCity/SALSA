import torch
from torch.utils.data import Dataset

class LWEDataset(Dataset):
    def __init__(self, LWE, base):
        self.A = LWE.A
        self.b = LWE.b
        self.q = LWE.q
        self.n = LWE.n
        self.base = base

        self.digits_per_int = self._digit_len(self.q, self.base) # how many base B digits are needed per value

        # unique tokens for separation (enoder) and start/end of sequence (decoder)
        self.SEP = base
        self.SOS = base + 1
        self.EOS = base + 2
        self.vocab_size = base + 3

    def _digit_len(self, q, base): # returns k the smallest amount of digits to uniquely represent each value 1,...,q
        k = 0
        val = 1
        while val<q:
            val *= base
            k+= 1
        return k

    def _encode_int(self, x): # converts x to a list of base B values with the most significant first
        digits = []
        for _ in range(self.digits_per_int):
            digits.append(x % self.base)
            x //= self.base
        return digits[::-1]

    def _decode_int(self, digits): # converts from base B back to base 10
        x = 0
        for digit in digits:
            x = x * self.base + int(digit)
        return x

    def __len__(self): # number of training samples
        return len(self.b)

    def __getitem__(self, idx): # how the model grabs a single sample
        row = self.A[idx]
        src = []
        for j,val in enumerate(row):
            src.extend(self._encode_int(val))
            if j<len(row)-1:
                src.append(self.SEP)
        tgt = [self.SOS] + self._encode_int(self.b[idx]) + [self.EOS]
        return (torch.tensor(src, dtype=torch.long), torch.tensor(tgt,dtype=torch.long),)