"""
Training Loop for Heart Sound Classification

Full training pipeline with:
- Multi-task training (Murmur + Outcome)
- Learning rate scheduling with warmup
- Early stopping
- Checkpointing
- TensorBoard logging
- Progress bars
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import time
import os

from .losses import MultiTaskLoss, create_loss_function
from .metrics import MetricTracker, EarlyStopping, print_metrics


class Trainer:
    """
    Training orchestrator for multi-task heart sound classification.
    
    Handles the complete training lifecycle including:
    - Training and validation loops
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: str = 'cuda',
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        murmur_class_weights: Optional[torch.Tensor] = None,
        outcome_class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            model: HeartSoundClassifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object with training parameters
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            murmur_class_weights: Class weights for murmur loss
            outcome_class_weights: Class weights for outcome loss
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path('checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir) if log_dir else Path('logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = create_loss_function(
            murmur_class_weights=murmur_class_weights,
            outcome_class_weights=outcome_class_weights,
            murmur_weight=config.training.murmur_loss_weight,
            outcome_weight=config.training.outcome_loss_weight,
            focal_gamma=2.0,
            label_smoothing=config.training.label_smoothing
        )
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.metrics = MetricTracker()
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='min'
        )
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999)
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup."""
        warmup_epochs = self.config.training.warmup_epochs
        total_epochs = self.config.training.num_epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        murmur_loss_sum = 0.0
        outcome_loss_sum = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}',
            leave=False
        )
        
        for batch in progress_bar:
            spectrograms, murmur_labels, outcome_labels, _ = batch
            
            spectrograms = spectrograms.to(self.device)
            targets = {
                'murmur': murmur_labels.to(self.device),
                'outcome': outcome_labels.to(self.device)
            }
            
            self.optimizer.zero_grad()
            
            outputs = self.model(spectrograms)
            
            losses = self.criterion(outputs, targets)
            loss = losses['total']
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            murmur_loss_sum += losses['murmur'].item()
            outcome_loss_sum += losses['outcome'].item()
            num_batches += 1
            
            self.metrics.update(outputs, targets)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        avg_murmur_loss = murmur_loss_sum / num_batches
        avg_outcome_loss = outcome_loss_sum / num_batches
        
        metrics = self.metrics.compute()
        metrics['loss'] = avg_loss
        metrics['murmur_loss'] = avg_murmur_loss
        metrics['outcome_loss'] = avg_outcome_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        murmur_loss_sum = 0.0
        outcome_loss_sum = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.val_loader,
            desc='Validation',
            leave=False
        )
        
        for batch in progress_bar:
            spectrograms, murmur_labels, outcome_labels, _ = batch
            
            spectrograms = spectrograms.to(self.device)
            targets = {
                'murmur': murmur_labels.to(self.device),
                'outcome': outcome_labels.to(self.device)
            }
            
            outputs = self.model(spectrograms)
            
            losses = self.criterion(outputs, targets)
            
            total_loss += losses['total'].item()
            murmur_loss_sum += losses['murmur'].item()
            outcome_loss_sum += losses['outcome'].item()
            num_batches += 1
            
            self.metrics.update(outputs, targets)
        
        avg_loss = total_loss / num_batches
        avg_murmur_loss = murmur_loss_sum / num_batches
        avg_outcome_loss = outcome_loss_sum / num_batches
        
        metrics = self.metrics.compute()
        metrics['loss'] = avg_loss
        metrics['murmur_loss'] = avg_murmur_loss
        metrics['outcome_loss'] = avg_outcome_loss
        
        return metrics
    
    def fit(self, num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Full training loop with validation and early stopping.
        
        Args:
            num_epochs: Number of epochs (uses config if not provided)
        
        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config.training.num_epochs
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_murmur_acc': [],
            'val_murmur_acc': [],
            'train_outcome_acc': [],
            'val_outcome_acc': [],
            'learning_rates': []
        }
        
        print(f"\n{'='*60}")
        print(f"Starting Training - {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            train_metrics = self.train_epoch()
            
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_murmur_acc'].append(train_metrics['murmur_accuracy'])
            history['val_murmur_acc'].append(val_metrics['murmur_accuracy'])
            history['train_outcome_acc'].append(train_metrics['outcome_accuracy'])
            history['val_outcome_acc'].append(val_metrics['outcome_accuracy'])
            history['learning_rates'].append(current_lr)
            
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr)
            
            epoch_time = time.time() - epoch_start
            self._print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, epoch_time)
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt', val_metrics)
                print(f"  💾 New best model saved! (val_loss: {val_metrics['loss']:.4f})")
            
            if self.early_stopping(val_metrics['loss']):
                print(f"\n⚠️ Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        self.save_checkpoint('final_model.pt', val_metrics)
        self.writer.close()
        
        return history
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float
    ):
        """Log metrics to TensorBoard."""
        self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        self.writer.add_scalar('Train/MurmurLoss', train_metrics['murmur_loss'], epoch)
        self.writer.add_scalar('Train/OutcomeLoss', train_metrics['outcome_loss'], epoch)
        self.writer.add_scalar('Train/MurmurAccuracy', train_metrics['murmur_accuracy'], epoch)
        self.writer.add_scalar('Train/OutcomeAccuracy', train_metrics['outcome_accuracy'], epoch)
        self.writer.add_scalar('Train/MurmurF1', train_metrics['murmur_f1'], epoch)
        self.writer.add_scalar('Train/OutcomeF1', train_metrics['outcome_f1'], epoch)
        
        self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        self.writer.add_scalar('Val/MurmurLoss', val_metrics['murmur_loss'], epoch)
        self.writer.add_scalar('Val/OutcomeLoss', val_metrics['outcome_loss'], epoch)
        self.writer.add_scalar('Val/MurmurAccuracy', val_metrics['murmur_accuracy'], epoch)
        self.writer.add_scalar('Val/OutcomeAccuracy', val_metrics['outcome_accuracy'], epoch)
        self.writer.add_scalar('Val/MurmurF1', val_metrics['murmur_f1'], epoch)
        self.writer.add_scalar('Val/OutcomeF1', val_metrics['outcome_f1'], epoch)
        
        self.writer.add_scalar('LearningRate', lr, epoch)
    
    def _print_epoch_summary(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Print epoch summary."""
        print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Murmur Acc: {train_metrics['murmur_accuracy']:.4f} | "
              f"Outcome Acc: {train_metrics['outcome_accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Murmur Acc: {val_metrics['murmur_accuracy']:.4f} | "
              f"Outcome Acc: {val_metrics['outcome_accuracy']:.4f}")
    
    def save_checkpoint(self, filename: str, metrics: Optional[Dict[str, float]] = None):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
            metrics: Optional metrics to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'metrics': metrics
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            filename: Checkpoint filename
        
        Returns:
            Checkpoint dictionary
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        
        return checkpoint


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Any,
    class_weights: Optional[Dict[str, torch.Tensor]] = None
) -> Trainer:
    """
    Factory function to create Trainer with proper setup.
    
    Args:
        model: HeartSoundClassifier model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        config: Configuration object
        class_weights: Dict from get_class_weights() with murmur and outcome weights
    
    Returns:
        Configured Trainer instance
    """
    device = config.device if hasattr(config, 'device') else 'cuda'
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠️ CUDA not available, using CPU")
    
    murmur_weights = None
    outcome_weights = None
    if class_weights:
        murmur_weights = class_weights.get('murmur_weights')
        outcome_weights = class_weights.get('outcome_weights')
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=str(config.paths.checkpoint_dir),
        log_dir=str(config.paths.log_dir),
        murmur_class_weights=murmur_weights,
        outcome_class_weights=outcome_weights
    )


if __name__ == "__main__":
    print("Trainer module loaded successfully")
    print("Use create_trainer() to instantiate a Trainer object")
