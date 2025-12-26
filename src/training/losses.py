"""
Multi-Task Loss Functions for Heart Sound Classification

Implements:
- FocalLoss: Handles class imbalance by down-weighting easy examples
- MultiTaskLoss: Combines murmur and outcome losses with configurable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(pt) = -α * (1 - pt)^γ * log(pt)
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights tensor (optional)
        label_smoothing: Label smoothing factor (default: 0.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        
        Returns:
            Focal loss value
        """
        num_classes = logits.size(-1)
        
        ce_loss = F.cross_entropy(
            logits, 
            targets, 
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining Murmur and Outcome classification losses.
    
    Total Loss = murmur_weight * MurmurLoss + outcome_weight * OutcomeLoss
    
    Args:
        murmur_weight: Weight for murmur classification loss
        outcome_weight: Weight for outcome classification loss
        murmur_class_weights: Class weights for murmur (handles imbalance)
        outcome_class_weights: Class weights for outcome (handles imbalance)
        focal_gamma: Gamma parameter for focal loss
        label_smoothing: Label smoothing factor
        use_focal_for_outcome: Whether to use focal loss for outcome too
    """
    
    def __init__(
        self,
        murmur_weight: float = 0.6,
        outcome_weight: float = 0.4,
        murmur_class_weights: Optional[torch.Tensor] = None,
        outcome_class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
        use_focal_for_outcome: bool = True
    ):
        super().__init__()
        self.murmur_weight = murmur_weight
        self.outcome_weight = outcome_weight
        
        self.murmur_loss_fn = FocalLoss(
            gamma=focal_gamma,
            alpha=murmur_class_weights,
            label_smoothing=label_smoothing
        )
        
        if use_focal_for_outcome:
            self.outcome_loss_fn = FocalLoss(
                gamma=focal_gamma,
                alpha=outcome_class_weights,
                label_smoothing=label_smoothing
            )
        else:
            self.outcome_loss_fn = nn.CrossEntropyLoss(
                weight=outcome_class_weights,
                label_smoothing=label_smoothing
            )
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs dict with 'murmur' and 'outcome' logits
            targets: Target labels dict with 'murmur' and 'outcome' labels
        
        Returns:
            Dict with 'total', 'murmur', and 'outcome' losses
        """
        murmur_logits = outputs['murmur']
        outcome_logits = outputs['outcome']
        
        murmur_targets = targets['murmur']
        outcome_targets = targets['outcome']
        
        murmur_loss = self.murmur_loss_fn(murmur_logits, murmur_targets)
        outcome_loss = self.outcome_loss_fn(outcome_logits, outcome_targets)
        
        total_loss = (
            self.murmur_weight * murmur_loss + 
            self.outcome_weight * outcome_loss
        )
        
        return {
            'total': total_loss,
            'murmur': murmur_loss,
            'outcome': outcome_loss
        }


def create_loss_function(
    murmur_class_weights: Optional[torch.Tensor] = None,
    outcome_class_weights: Optional[torch.Tensor] = None,
    murmur_weight: float = 0.6,
    outcome_weight: float = 0.4,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.1
) -> MultiTaskLoss:
    """
    Factory function to create MultiTaskLoss with class weights.
    
    Args:
        murmur_class_weights: Weights for murmur classes (from get_class_weights)
        outcome_class_weights: Weights for outcome classes (from get_class_weights)
        murmur_weight: Weight for murmur loss in total loss
        outcome_weight: Weight for outcome loss in total loss
        focal_gamma: Gamma for focal loss
        label_smoothing: Label smoothing factor
    
    Returns:
        Configured MultiTaskLoss instance
    """
    return MultiTaskLoss(
        murmur_weight=murmur_weight,
        outcome_weight=outcome_weight,
        murmur_class_weights=murmur_class_weights,
        outcome_class_weights=outcome_class_weights,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing
    )


if __name__ == "__main__":
    print("Testing loss functions...")
    
    batch_size = 8
    murmur_logits = torch.randn(batch_size, 2)
    outcome_logits = torch.randn(batch_size, 2)
    murmur_targets = torch.randint(0, 2, (batch_size,))
    outcome_targets = torch.randint(0, 2, (batch_size,))
    
    outputs = {'murmur': murmur_logits, 'outcome': outcome_logits}
    targets = {'murmur': murmur_targets, 'outcome': outcome_targets}
    
    loss_fn = create_loss_function()
    losses = loss_fn(outputs, targets)
    
    print(f"Total loss: {losses['total'].item():.4f}")
    print(f"Murmur loss: {losses['murmur'].item():.4f}")
    print(f"Outcome loss: {losses['outcome'].item():.4f}")
    print("\n✓ Loss functions working correctly!")
