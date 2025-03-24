import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import glob
import numpy as np
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

class FocalLoss(nn.Module):
    """Focal loss: helps handle class imbalance by reducing the loss contribution from easy examples and focusing on harder ones"""
    def __init__(self, alpha=1, gamma=2, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # Compute Binary Cross-Entropy loss with logits
        # 'pos_weight' adjusts the weight of positive samples to address class imbalance
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        
        # Apply sigmoid to convert logits to probabilities between 0 and 1
        probs = torch.sigmoid(inputs)
        
        # Select the probability of the true class
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute the focal loss term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # Return the mean focal loss over the batch
        return focal_loss.mean()

def freeze_backbone(model):
    """Freeze the backbone of the model"""
    for param in model.model.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    """Unfreeze the backbone of the model"""
    for param in model.model.parameters():
        param.requires_grad = True

def build_optimizer(config, model_params):
    """Build optimizer based on configuration."""
    optimizer_type = config.optimizer["type"]
    optimizer_params = config.optimizer["params"]
    if optimizer_type == "Adam":
        return Adam(model_params, **optimizer_params)
    elif optimizer_type == "SGD":
        return SGD(model_params, **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def build_scheduler(config, optimizer, train_loader_length, current_epoch, total_epochs):
    """Build scheduler based on configuration."""
    scheduler_type = config.training["scheduler"]["type"]
    T_max_factor = config.training["scheduler"].get("T_max_factor", 1)

    if scheduler_type == "CosineAnnealingLR":
        T_max = int(T_max_factor * (total_epochs - current_epoch) * train_loader_length)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

def calculate_positive_weight(data_dir, device, block_filaname):
    """Calculate positive weight for handling class imbalance."""
    label_list = []
    block_files = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.startswith(block_filaname.split("\{")[0])
            and f.endswith(block_filaname.split("\{")[1])
    ]

    if len(block_files) == 0:
        raise ValueError("No label blocks found in directory. Check path and preprocessing.")

    for block_file in block_files:
        with np.load(block_file) as block:
            labels = block["labels"]
            label_list.append(labels)

    all_labels = np.concatenate(label_list)
    counts = {
        0: (all_labels == 0).sum(),
        1: (all_labels == 1).sum(),
    }

    if counts[1] > 0:
        pos_weight = torch.tensor([counts[0] / counts[1]], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)

    return pos_weight