import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW, Optimizer

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model

    
def get_adamw_optimizer(model, lr, weight_decay, exclude_params=['bias', 'LayerNorm']):
    parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in exclude_params], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if n in exclude_params], 'weight_decay': 0.0}  # No weight decay for certain parameters
    ]
    optimizer = AdamW(parameters, lr=lr)
    return optimizer
