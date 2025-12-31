"""
Training Module for Heart Sound Classification

Components:
- losses: Multi-task loss functions with Focal Loss
- metrics: MetricTracker for evaluation
- trainer: Full training loop with checkpointing
"""

from .losses import FocalLoss, MultiTaskLoss, create_loss_function
from .metrics import MetricTracker, EarlyStopping, print_metrics
from .trainer import Trainer, create_trainer

__all__ = [
    'FocalLoss',
    'MultiTaskLoss',
    'create_loss_function',
    'MetricTracker',
    'EarlyStopping',
    'print_metrics',
    'Trainer',
    'create_trainer'
]
