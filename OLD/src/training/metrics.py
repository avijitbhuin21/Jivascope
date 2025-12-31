"""
Evaluation Metrics for Heart Sound Classification

Implements MetricTracker to compute:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- Per-class metrics for both Murmur and Outcome tasks
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


class MetricTracker:
    """
    Tracks predictions and computes classification metrics for multi-task learning.
    
    Accumulates predictions over batches and computes metrics on demand.
    """
    
    MURMUR_CLASSES = ['Absent', 'Present']
    OUTCOME_CLASSES = ['Normal', 'Abnormal']
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated predictions and targets."""
        self.murmur_preds: List[int] = []
        self.murmur_targets: List[int] = []
        self.murmur_probs: List[np.ndarray] = []
        
        self.outcome_preds: List[int] = []
        self.outcome_targets: List[int] = []
        self.outcome_probs: List[np.ndarray] = []
    
    def update(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ):
        """
        Update tracker with batch predictions and targets.
        
        Args:
            outputs: Model outputs with 'murmur' and 'outcome' logits
            targets: Ground truth labels
        """
        murmur_logits = outputs['murmur'].detach()
        outcome_logits = outputs['outcome'].detach()
        
        murmur_probs = torch.softmax(murmur_logits, dim=1).cpu().numpy()
        murmur_pred = murmur_logits.argmax(dim=1).cpu().numpy()
        self.murmur_preds.extend(murmur_pred.tolist())
        self.murmur_targets.extend(targets['murmur'].cpu().numpy().tolist())
        self.murmur_probs.extend(murmur_probs)
        
        outcome_probs = torch.softmax(outcome_logits, dim=1).cpu().numpy()
        outcome_pred = outcome_logits.argmax(dim=1).cpu().numpy()
        self.outcome_preds.extend(outcome_pred.tolist())
        self.outcome_targets.extend(targets['outcome'].cpu().numpy().tolist())
        self.outcome_probs.extend(outcome_probs)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.
        
        Returns:
            Dictionary with all computed metrics
        """
        results = {}
        
        results['murmur_accuracy'] = accuracy_score(
            self.murmur_targets, self.murmur_preds
        )
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.murmur_targets, 
            self.murmur_preds, 
            average='weighted',
            zero_division=0
        )
        results['murmur_precision'] = prec
        results['murmur_recall'] = rec
        results['murmur_f1'] = f1
        
        results['outcome_accuracy'] = accuracy_score(
            self.outcome_targets, self.outcome_preds
        )
        
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.outcome_targets, 
            self.outcome_preds, 
            average='weighted',
            zero_division=0
        )
        results['outcome_precision'] = prec
        results['outcome_recall'] = rec
        results['outcome_f1'] = f1
        
        results['combined_accuracy'] = (
            results['murmur_accuracy'] + results['outcome_accuracy']
        ) / 2
        results['combined_f1'] = (
            results['murmur_f1'] + results['outcome_f1']
        ) / 2
        
        return results
    
    def compute_confusion_matrices(self) -> Dict[str, np.ndarray]:
        """
        Compute confusion matrices for both tasks.
        
        Returns:
            Dictionary with 'murmur' and 'outcome' confusion matrices
        """
        return {
            'murmur': confusion_matrix(
                self.murmur_targets, 
                self.murmur_preds,
                labels=[0, 1]
            ),
            'outcome': confusion_matrix(
                self.outcome_targets, 
                self.outcome_preds,
                labels=[0, 1]
            )
        }
    
    def get_classification_reports(self) -> Dict[str, str]:
        """
        Get detailed classification reports for both tasks.
        
        Returns:
            Dictionary with formatted classification reports
        """
        return {
            'murmur': classification_report(
                self.murmur_targets,
                self.murmur_preds,
                target_names=self.MURMUR_CLASSES,
                zero_division=0
            ),
            'outcome': classification_report(
                self.outcome_targets,
                self.outcome_preds,
                target_names=self.OUTCOME_CLASSES,
                zero_division=0
            )
        }
    
    def get_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get per-class precision, recall, F1 for both tasks.
        
        Returns:
            Nested dictionary with per-class metrics
        """
        results = {}
        
        prec, rec, f1, sup = precision_recall_fscore_support(
            self.murmur_targets,
            self.murmur_preds,
            average=None,
            zero_division=0
        )
        results['murmur'] = {
            cls: {'precision': p, 'recall': r, 'f1': f, 'support': int(s)}
            for cls, p, r, f, s in zip(self.MURMUR_CLASSES, prec, rec, f1, sup)
        }
        
        prec, rec, f1, sup = precision_recall_fscore_support(
            self.outcome_targets,
            self.outcome_preds,
            average=None,
            zero_division=0
        )
        results['outcome'] = {
            cls: {'precision': p, 'recall': r, 'f1': f, 'support': int(s)}
            for cls, p, r, f, s in zip(self.OUTCOME_CLASSES, prec, rec, f1, sup)
        }
        
        return results
    
    def __len__(self) -> int:
        """Return number of samples tracked."""
        return len(self.murmur_preds)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like accuracy
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (loss or metric)
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def print_metrics(metrics: Dict[str, float], epoch: Optional[int] = None):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names and values
        epoch: Optional epoch number for display
    """
    header = f"Epoch {epoch}" if epoch is not None else "Metrics"
    print(f"\n{'='*50}")
    print(f"{header:^50}")
    print(f"{'='*50}")
    
    print("\n📊 Murmur Classification:")
    print(f"  Accuracy:  {metrics['murmur_accuracy']:.4f}")
    print(f"  Precision: {metrics['murmur_precision']:.4f}")
    print(f"  Recall:    {metrics['murmur_recall']:.4f}")
    print(f"  F1-Score:  {metrics['murmur_f1']:.4f}")
    
    print("\n🏥 Outcome Classification:")
    print(f"  Accuracy:  {metrics['outcome_accuracy']:.4f}")
    print(f"  Precision: {metrics['outcome_precision']:.4f}")
    print(f"  Recall:    {metrics['outcome_recall']:.4f}")
    print(f"  F1-Score:  {metrics['outcome_f1']:.4f}")
    
    print("\n🎯 Combined:")
    print(f"  Accuracy:  {metrics['combined_accuracy']:.4f}")
    print(f"  F1-Score:  {metrics['combined_f1']:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("Testing MetricTracker...")
    
    tracker = MetricTracker()
    
    for _ in range(5):
        batch_size = 8
        outputs = {
            'murmur': torch.randn(batch_size, 2),
            'outcome': torch.randn(batch_size, 2)
        }
        targets = {
            'murmur': torch.randint(0, 2, (batch_size,)),
            'outcome': torch.randint(0, 2, (batch_size,))
        }
        tracker.update(outputs, targets)
    
    metrics = tracker.compute()
    print_metrics(metrics, epoch=1)
    
    cm = tracker.compute_confusion_matrices()
    print("Confusion Matrices:")
    print(f"Murmur:\n{cm['murmur']}")
    print(f"Outcome:\n{cm['outcome']}")
    
    print("\n✓ MetricTracker working correctly!")
