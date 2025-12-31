# Step 4: Training Pipeline

## Objective
Implement the complete training loop with multi-task loss, class balancing, and training visualization.

## Prerequisites
- Step 3 completed (model architecture ready)
- Data pipeline from Step 2

---

## Implementation Details

### 4.1 Multi-Task Loss Function

Create `src/training/losses.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    """
    Weighted combination of Murmur (3-class) and Outcome (binary) losses.
    
    Total Loss = α * MurmurLoss + β * OutcomeLoss
    
    Uses Focal Loss to handle class imbalance.
    """
    
    def __init__(self, 
                 murmur_weight: float = 0.5,
                 outcome_weight: float = 0.5,
                 murmur_class_weights: torch.Tensor = None,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.murmur_weight = murmur_weight
        self.outcome_weight = outcome_weight
        self.focal_gamma = focal_gamma
        
        # Focal Loss for murmur (handles class imbalance)
        self.murmur_class_weights = murmur_class_weights
        
    def focal_loss(self, logits, targets, class_weights=None):
        """Focal Loss for handling class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
        return focal_loss.mean()
    
    def forward(self, outputs, targets):
        murmur_logits = outputs['murmur']
        outcome_logits = outputs['outcome']
        
        murmur_targets = targets['murmur']
        outcome_targets = targets['outcome'].float()
        
        # Murmur: Focal Loss (3-class)
        murmur_loss = self.focal_loss(murmur_logits, murmur_targets, self.murmur_class_weights)
        
        # Outcome: Binary Cross Entropy
        outcome_loss = F.binary_cross_entropy_with_logits(
            outcome_logits.squeeze(), outcome_targets
        )
        
        total_loss = self.murmur_weight * murmur_loss + self.outcome_weight * outcome_loss
        
        return {
            'total': total_loss,
            'murmur': murmur_loss,
            'outcome': outcome_loss
        }
```

### 4.2 Evaluation Metrics

Create `src/training/metrics.py`:

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class MetricTracker:
    """Track and compute classification metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.murmur_preds = []
        self.murmur_targets = []
        self.outcome_preds = []
        self.outcome_targets = []
    
    def update(self, outputs, targets):
        # Murmur: argmax of logits
        murmur_pred = outputs['murmur'].argmax(dim=1).cpu().numpy()
        self.murmur_preds.extend(murmur_pred)
        self.murmur_targets.extend(targets['murmur'].cpu().numpy())
        
        # Outcome: sigmoid > 0.5
        outcome_pred = (torch.sigmoid(outputs['outcome']) > 0.5).int().cpu().numpy()
        self.outcome_preds.extend(outcome_pred.flatten())
        self.outcome_targets.extend(targets['outcome'].cpu().numpy())
    
    def compute(self):
        results = {}
        
        # Murmur metrics
        results['murmur_accuracy'] = accuracy_score(self.murmur_targets, self.murmur_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.murmur_targets, self.murmur_preds, average='weighted'
        )
        results['murmur_precision'] = prec
        results['murmur_recall'] = rec
        results['murmur_f1'] = f1
        
        # Outcome metrics
        results['outcome_accuracy'] = accuracy_score(self.outcome_targets, self.outcome_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.outcome_targets, self.outcome_preds, average='binary'
        )
        results['outcome_precision'] = prec
        results['outcome_recall'] = rec
        results['outcome_f1'] = f1
        
        return results
```

### 4.3 Training Configuration

Create `configs/default.yaml`:

```yaml
# Training hyperparameters
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 15
  
# Optimizer
optimizer:
  name: AdamW
  betas: [0.9, 0.999]
  
# Scheduler
scheduler:
  name: CosineAnnealingLR
  T_max: 100
  eta_min: 0.00001
  
# Model
model:
  backbone: efficientnet_b0
  pretrained: true
  hidden_dim: 512
  dropout: 0.3
  
# Loss
loss:
  murmur_weight: 0.5
  outcome_weight: 0.5
  focal_gamma: 2.0
  
# Data
data:
  sample_rate: 4000
  target_duration: 5.0  # seconds
  n_mels: 128
  n_fft: 512
  hop_length: 128
```

### 4.4 Trainer Class

Create `src/training/trainer.py`:

```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss()
        self.metrics = MetricTracker()
        
    def train_epoch(self):
        """Single training epoch."""
        pass
    
    def validate(self):
        """Validation loop."""
        pass
    
    def fit(self):
        """Full training loop with early stopping."""
        pass
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        pass
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        pass
```

### 4.5 Colab Training Notebook

Create `notebooks/02_training.ipynb`:

1. Mount Google Drive
2. Install dependencies
3. Load data
4. Initialize model
5. Train with GPU
6. Save best checkpoint to Drive

---

## Research Areas
- Optimal learning rate finder (LR range test)
- Mixed precision training (fp16) for faster training
- Gradient accumulation if batch size limited by memory
- Label smoothing for better generalization

---

## Expected Outcome
- `losses.py` with multi-task focal loss
- `metrics.py` with metric tracking
- `trainer.py` with full training loop
- `configs/default.yaml` with hyperparameters
- TensorBoard logging integration

---

## Estimated Effort
- **1-2 days** for training pipeline implementation

---

## Dependencies
- Step 2: Data pipeline
- Step 3: Model architecture
