# Step 4: Training Pipeline

## Objective
Implement training loop with early stopping, class weighting, and checkpointing.

## Tasks

### 4.1 Training Configuration
```python
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'epochs': 100,
    'early_stopping_patience': 15,
    'scheduler': 'cosine_annealing',
    'gradient_clip': 1.0
}
```

### 4.2 Loss Function (Multi-Label)
```python
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight_heart=1.0, pos_weight_murmur=2.5):
        super().__init__()
        self.pos_weights = torch.tensor([pos_weight_heart, pos_weight_murmur])
    
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=self.pos_weights.to(logits.device)
        )
```

### 4.3 Trainer Class
```python
class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=config['lr'])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        self.criterion = WeightedBCELoss()
        self.early_stopping = EarlyStopping(patience=15)
    
    def train_epoch(self):
        self.model.train()
        for batch in train_loader:
            features, labels = batch
            logits, _ = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
    
    def validate(self):
        self.model.eval()
        # Calculate accuracy, precision, recall, F1
```

### 4.4 Metrics Tracking
Track per-class metrics:
- Heart Sound: Accuracy, Precision, Recall, F1
- Murmur: Accuracy, Sensitivity, Specificity, F1

### 4.5 Checkpointing
```python
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, path)
```

## Deliverables
- [ ] `model/trainer.py` - Training loop
- [ ] `model/losses.py` - Loss functions
- [ ] `train.py` - Main training script

## Estimated Time
~3-4 hours
