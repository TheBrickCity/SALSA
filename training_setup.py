import torch
from torch.optim.lr_scheduler import LambdaLR

def make_optimizer_and_scheduler(model, learning_rate, warmup_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    def lr_lambda(step):
        if step < warmup_steps:
            return step/warmup_steps
        return 1.0
    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler