import torch
import torch.nn as nn

def dice_loss(outputs, targets, smooth=1e-6):
    outputs = torch.sigmoid(outputs)
    intersection = (outputs * targets).sum(dim=(1,2,3))
    union = outputs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    loss = 1 - ((2. * intersection + smooth) / (union + smooth))
    return loss.mean()

def combined_loss(outputs, targets, smooth=1e-6, alpha=0.5):
    bce = nn.BCEWithLogitsLoss()(outputs, targets)
    dice = dice_loss(outputs, targets, smooth)
    return alpha * bce + (1 - alpha) * dice
