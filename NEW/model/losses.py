"""
Loss Functions for LightCardiacNet

Custom loss functions for multi-label heart sound classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    def __init__(
        self,
        pos_weight_heart: float = 1.0,
        pos_weight_murmur: float = 2.5
    ):
        super().__init__()
        self.register_buffer(
            'pos_weights',
            torch.tensor([pos_weight_heart, pos_weight_murmur])
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weights.to(logits.device)
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        pos_weight_heart: float = 1.0,
        pos_weight_murmur: float = 2.5
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce = WeightedBCELoss(pos_weight_heart, pos_weight_murmur)
        self.focal = FocalLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss


def create_loss_function(
    loss_type: str = 'weighted_bce',
    pos_weight_heart: float = 1.0,
    pos_weight_murmur: float = 2.5
) -> nn.Module:
    if loss_type == 'weighted_bce':
        return WeightedBCELoss(pos_weight_heart, pos_weight_murmur)
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'combined':
        return CombinedLoss(pos_weight_heart=pos_weight_heart, pos_weight_murmur=pos_weight_murmur)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
