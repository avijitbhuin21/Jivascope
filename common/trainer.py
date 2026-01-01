"""
Training Utilities for Heart Sound Classification Models.

Provides base trainer class with early stopping, checkpointing, and metrics tracking.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Base trainer class for heart sound classification models."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('epochs', 100)
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        
        self.early_stopping = EarlyStopping(
            patience=config.get('early_stopping_patience', 15),
            mode='min'
        )
        
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'val_auc': []
        }
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        avg_acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        
        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy_score(all_labels.flatten(), all_preds.flatten()),
            'heart_accuracy': accuracy_score(all_labels[:, 0], all_preds[:, 0]),
            'murmur_accuracy': accuracy_score(all_labels[:, 1], all_preds[:, 1]),
            'murmur_precision': precision_score(all_labels[:, 1], all_preds[:, 1], zero_division=0),
            'murmur_recall': recall_score(all_labels[:, 1], all_preds[:, 1], zero_division=0),
            'murmur_f1': f1_score(all_labels[:, 1], all_preds[:, 1], zero_division=0),
        }
        
        try:
            metrics['murmur_auc'] = roc_auc_score(all_labels[:, 1], all_probs[:, 1])
        except ValueError:
            metrics['murmur_auc'] = 0.0
        
        return metrics
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict, is_best_loss: bool = False, is_best_acc: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'config': self.config
        }
        
        if is_best_loss:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        if is_best_acc:
            best_acc_path = os.path.join(self.checkpoint_dir, 'best_acc_model.pt')
            torch.save(checkpoint, best_acc_path)
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """Full training loop."""
        epochs = epochs or self.config.get('epochs', 100)
        save_every = self.config.get('save_every', 5)
        
        start_time = time.time()
        epoch_times = []
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)
            
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['loss'])
            
            is_best_loss = val_metrics['loss'] < self.best_val_loss
            is_best_acc = val_metrics['accuracy'] > self.best_val_acc
            
            if is_best_loss:
                self.best_val_loss = val_metrics['loss']
            if is_best_acc:
                self.best_val_acc = val_metrics['accuracy']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['murmur_f1'])
            self.history['val_auc'].append(val_metrics['murmur_auc'])
            
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            eta_minutes = (avg_epoch_time * remaining_epochs) / 60
            
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s, ETA: {eta_minutes:.1f}min)")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Murmur - P: {val_metrics['murmur_precision']:.4f}, R: {val_metrics['murmur_recall']:.4f}, F1: {val_metrics['murmur_f1']:.4f}, AUC: {val_metrics['murmur_auc']:.4f}")
            
            if is_best_loss or is_best_acc:
                self.save_checkpoint(epoch, val_metrics, is_best_loss=is_best_loss, is_best_acc=is_best_acc)
                if is_best_loss:
                    print("  ✓ New best loss model saved! (best_model.pt)")
                if is_best_acc:
                    print(f"  ✓ New best accuracy model saved! ({val_metrics['accuracy']:.4f}) (best_acc_model.pt)")
            
            if (epoch + 1) % save_every == 0:
                periodic_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'config': self.config
                }
                torch.save(checkpoint, periodic_path)
                print(f"  📁 Periodic checkpoint saved (epoch {epoch+1})")
            
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Avg epoch time: {np.mean(epoch_times):.1f}s")
        print(f"  Best Val Loss: {self.best_val_loss:.4f}")
        print(f"  Best Val Accuracy: {self.best_val_acc:.4f}")
        
        return self.history
