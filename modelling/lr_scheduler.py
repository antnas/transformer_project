import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

class TransformerLR(LRScheduler):
    """
    Learning rate scheduler for the Transformer model.

    This scheduler adjusts the learning rate based on the current step count and the model's dimensionality (`d_model`).
    It uses a warmup strategy to gradually increase the learning rate during the initial training steps.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        d_model (int): The dimensionality of the model.
        warmup_steps (int): The number of warmup steps.
        last_epoch (int, optional): The index of the last epoch. Default is -1.
        verbose (bool, optional): If True, prints a message for each update. Default is False.

    Attributes:
        d_model (int): The dimensionality of the model.
        warmup_steps (int): The number of warmup steps.

    Methods:
        get_lr(): Calculates the learning rate based on the current step count and the model's dimensionality.

    Example:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = TransformerLR(optimizer, d_model=512, warmup_steps=1000)
        for epoch in range(num_epochs):
            train(...)
            scheduler.step()
    """
    def __init__(self, optimizer, d_model, warmup_steps, last_epoch=-1, verbose=False):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        lr = self.d_model**(-0.5) * min(self._step_count**(-0.5), self._step_count * self.warmup_steps**(-1.5))
        return [base_lr * lr for base_lr in self.base_lrs]
    
def get_adamw_optimizer(model, lr, weight_decay, exclude_params=['bias', 'LayerNorm']):
    parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in exclude_params], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if n in exclude_params], 'weight_decay': 0.0}  # No weight decay for certain parameters
    ]
    optimizer = AdamW(parameters, lr=lr)
    return optimizer