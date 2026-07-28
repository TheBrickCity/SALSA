import torch
from torch.utils.data import Dataset

class LWEDataset(Dataset):
    def __init__(self, A, b, base):
        self.A = torch.tensor(A, dtype=torch.long)
        self.b = torch.tensor(b, dtype=torch.long)
        self.base = base
        self.SEP = base

    def __len__(self):
        return self.A.shape[0]

    # add getitem