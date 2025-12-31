"""
Unit Tests for Training Pipeline

Tests for:
- Loss functions (FocalLoss, MultiTaskLoss)
- MetricTracker
- EarlyStopping
- Trainer initialization
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.losses import FocalLoss, MultiTaskLoss, create_loss_function
from src.training.metrics import MetricTracker, EarlyStopping
from src.utils.config import get_default_config


class TestFocalLoss:
    """Tests for FocalLoss class."""
    
    def test_focal_loss_shape(self):
        """Test FocalLoss returns scalar."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.shape == torch.Size([])
        assert loss.item() > 0
    
    def test_focal_loss_with_class_weights(self):
        """Test FocalLoss with class weights."""
        weights = torch.tensor([1.0, 2.0])
        loss_fn = FocalLoss(gamma=2.0, alpha=weights)
        
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.shape == torch.Size([])
    
    def test_focal_loss_gamma_effect(self):
        """Test that higher gamma reduces easy example contribution."""
        logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
        targets = torch.tensor([0, 1])
        
        loss_gamma0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_gamma2 = FocalLoss(gamma=2.0)(logits, targets)
        
        assert loss_gamma2 <= loss_gamma0
    
    def test_focal_loss_reduction_none(self):
        """Test FocalLoss with no reduction."""
        loss_fn = FocalLoss(gamma=2.0, reduction='none')
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.shape == (8,)
    
    def test_focal_loss_label_smoothing(self):
        """Test FocalLoss with label smoothing."""
        loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
        logits = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert loss.shape == torch.Size([])


class TestMultiTaskLoss:
    """Tests for MultiTaskLoss class."""
    
    def test_multitask_loss_output_format(self):
        """Test MultiTaskLoss returns correct dictionary."""
        loss_fn = MultiTaskLoss()
        
        outputs = {
            'murmur': torch.randn(8, 2),
            'outcome': torch.randn(8, 2)
        }
        targets = {
            'murmur': torch.randint(0, 2, (8,)),
            'outcome': torch.randint(0, 2, (8,))
        }
        
        losses = loss_fn(outputs, targets)
        
        assert 'total' in losses
        assert 'murmur' in losses
        assert 'outcome' in losses
        assert all(isinstance(v, torch.Tensor) for v in losses.values())
    
    def test_multitask_loss_weights(self):
        """Test that loss weights are applied correctly."""
        loss_fn = MultiTaskLoss(murmur_weight=0.8, outcome_weight=0.2)
        
        outputs = {
            'murmur': torch.randn(8, 2),
            'outcome': torch.randn(8, 2)
        }
        targets = {
            'murmur': torch.randint(0, 2, (8,)),
            'outcome': torch.randint(0, 2, (8,))
        }
        
        losses = loss_fn(outputs, targets)
        
        expected_total = 0.8 * losses['murmur'] + 0.2 * losses['outcome']
        assert torch.allclose(losses['total'], expected_total, rtol=1e-5)
    
    def test_create_loss_function(self):
        """Test factory function."""
        murmur_weights = torch.tensor([1.0, 1.5])
        outcome_weights = torch.tensor([1.0, 1.2])
        
        loss_fn = create_loss_function(
            murmur_class_weights=murmur_weights,
            outcome_class_weights=outcome_weights,
            murmur_weight=0.6,
            outcome_weight=0.4
        )
        
        assert isinstance(loss_fn, MultiTaskLoss)
        assert loss_fn.murmur_weight == 0.6
        assert loss_fn.outcome_weight == 0.4


class TestMetricTracker:
    """Tests for MetricTracker class."""
    
    def test_metric_tracker_update(self):
        """Test MetricTracker accumulates predictions."""
        tracker = MetricTracker()
        
        outputs = {
            'murmur': torch.randn(8, 2),
            'outcome': torch.randn(8, 2)
        }
        targets = {
            'murmur': torch.randint(0, 2, (8,)),
            'outcome': torch.randint(0, 2, (8,))
        }
        
        tracker.update(outputs, targets)
        
        assert len(tracker) == 8
        assert len(tracker.murmur_preds) == 8
        assert len(tracker.outcome_preds) == 8
    
    def test_metric_tracker_compute(self):
        """Test MetricTracker computes all metrics."""
        tracker = MetricTracker()
        
        for _ in range(5):
            outputs = {
                'murmur': torch.randn(8, 2),
                'outcome': torch.randn(8, 2)
            }
            targets = {
                'murmur': torch.randint(0, 2, (8,)),
                'outcome': torch.randint(0, 2, (8,))
            }
            tracker.update(outputs, targets)
        
        metrics = tracker.compute()
        
        expected_keys = [
            'murmur_accuracy', 'murmur_precision', 'murmur_recall', 'murmur_f1',
            'outcome_accuracy', 'outcome_precision', 'outcome_recall', 'outcome_f1',
            'combined_accuracy', 'combined_f1'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert 0 <= metrics[key] <= 1
    
    def test_metric_tracker_reset(self):
        """Test MetricTracker reset clears all data."""
        tracker = MetricTracker()
        
        outputs = {
            'murmur': torch.randn(8, 2),
            'outcome': torch.randn(8, 2)
        }
        targets = {
            'murmur': torch.randint(0, 2, (8,)),
            'outcome': torch.randint(0, 2, (8,))
        }
        tracker.update(outputs, targets)
        
        assert len(tracker) == 8
        
        tracker.reset()
        
        assert len(tracker) == 0
        assert len(tracker.murmur_preds) == 0
    
    def test_metric_tracker_confusion_matrices(self):
        """Test confusion matrix computation."""
        tracker = MetricTracker()
        
        outputs = {
            'murmur': torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0]]),
            'outcome': torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0]])
        }
        targets = {
            'murmur': torch.tensor([0, 1, 1, 0]),
            'outcome': torch.tensor([0, 1, 0, 1])
        }
        tracker.update(outputs, targets)
        
        cm = tracker.compute_confusion_matrices()
        
        assert 'murmur' in cm
        assert 'outcome' in cm
        assert cm['murmur'].shape == (2, 2)
        assert cm['outcome'].shape == (2, 2)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        tracker = MetricTracker()
        
        outputs = {
            'murmur': torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0]]),
            'outcome': torch.tensor([[10.0, -10.0], [-10.0, 10.0], [10.0, -10.0], [-10.0, 10.0]])
        }
        targets = {
            'murmur': torch.tensor([0, 1, 0, 1]),
            'outcome': torch.tensor([0, 1, 0, 1])
        }
        tracker.update(outputs, targets)
        
        metrics = tracker.compute()
        
        assert metrics['murmur_accuracy'] == 1.0
        assert metrics['outcome_accuracy'] == 1.0


class TestEarlyStopping:
    """Tests for EarlyStopping class."""
    
    def test_early_stopping_min_mode(self):
        """Test early stopping in 'min' mode (for loss)."""
        es = EarlyStopping(patience=3, mode='min')
        
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.85)
        assert not es(0.9)
        assert not es(0.9)
        assert es(0.9)
    
    def test_early_stopping_max_mode(self):
        """Test early stopping in 'max' mode (for accuracy)."""
        es = EarlyStopping(patience=3, mode='max')
        
        assert not es(0.5)
        assert not es(0.6)
        assert not es(0.7)
        assert not es(0.65)
        assert not es(0.65)
        assert es(0.65)
    
    def test_early_stopping_reset(self):
        """Test early stopping reset."""
        es = EarlyStopping(patience=2, mode='min')
        
        es(1.0)
        es(1.1)
        es(1.1)
        es(1.1)
        
        assert es.early_stop
        
        es.reset()
        
        assert not es.early_stop
        assert es.counter == 0
        assert es.best_score is None
    
    def test_early_stopping_min_delta(self):
        """Test early stopping with min_delta threshold."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        assert not es(1.0)
        assert not es(0.99)
        assert not es(0.99)
        assert not es(0.98)


class TestTrainerIntegration:
    """Integration tests for Trainer class."""
    
    def test_trainer_initialization(self):
        """Test Trainer can be initialized."""
        from src.training.trainer import Trainer
        from src.models import create_model
        
        config = get_default_config()
        model = create_model(pretrained=False, device='cpu')
        
        dummy_dataset = torch.utils.data.TensorDataset(
            torch.randn(32, 3, 128, 313),
            torch.randint(0, 2, (32,)),
            torch.randint(0, 2, (32,)),
        )
        
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 32
            def __getitem__(self, idx):
                return (
                    torch.randn(3, 128, 313),
                    torch.randint(0, 2, (1,)).item(),
                    torch.randint(0, 2, (1,)).item(),
                    f"patient_{idx}"
                )
        
        train_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
        val_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
