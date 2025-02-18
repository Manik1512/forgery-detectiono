import torch
import torch.nn.functional as F

def dice_loss(pred, target, epsilon=1e-6):
    """Computes the Dice Loss."""
    intersection = 2 * (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (intersection / torch.max(union, torch.tensor(epsilon)))

def focal_loss(pred, target, gamma=2.0):
    """Computes the Focal Loss."""
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)  # Probabilities
    loss = (1 - pt) ** gamma * bce
    return loss.mean()

def total_loss(pred, target, lambda_factor=1.0, gamma=2.0):
    """Computes the combined Dice and Focal Loss."""
    return dice_loss(pred, target) + lambda_factor * focal_loss(pred, target, gamma)

