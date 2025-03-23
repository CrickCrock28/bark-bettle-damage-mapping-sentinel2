import torch.nn as nn
import torch.nn.functional as F
import torch

# Focal Loss: helps handle class imbalance by reducing the loss contribution from easy examples and focusing on harder ones
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

def freeze_backbone(model):
    for param in model.model.parameters():
        param.requires_grad = False

def unfreeze_backbone(model):
    for param in model.model.parameters():
        param.requires_grad = True
