import torch
import torch.nn as nn
import LWE_Sample_Manager, dataset, model
from torch.utils.data import DataLoader

lwe = LWE_Sample_Manager.LWE(10, 10, 251, 2, 3, 10)
lwe.generate()

integer_base = 81
ds = dataset.LWEDataset(lwe, integer_base)
