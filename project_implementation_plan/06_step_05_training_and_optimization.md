# Step 5: Training Execution & Model Optimization

## Objective
Train the model on Google Colab Pro, monitor performance, and optimize for 95%+ accuracy.

## Prerequisites
- Steps 1-4 completed (full training pipeline ready)
- Google Colab Pro account

---

## Implementation Details

### 5.1 Initial Training Run

Run baseline training:
- **Epochs**: 50-100
- **Batch size**: 32
- **Learning rate**: 1e-3 with cosine annealing
- **Backbone**: EfficientNet-B0 (pretrained)

Monitor:
- Training/validation loss curves
- Murmur accuracy (3-class)
- Outcome accuracy (binary)
- Check for overfitting (val loss starts increasing)

### 5.2 Hyperparameter Tuning

If baseline doesn't reach 95%, tune:

| Hyperparameter | Range to Try |
|----------------|--------------|
| Learning rate | 1e-4, 5e-4, 1e-3, 5e-3 |
| Batch size | 16, 32, 64 |
| Dropout | 0.2, 0.3, 0.5 |
| Hidden dim | 256, 512, 1024 |
| Backbone | ResNet18, ResNet34, EfficientNet-B2 |
| Focal gamma | 1.0, 2.0, 3.0 |

### 5.3 Advanced Optimization Techniques

If accuracy still insufficient:

1. **Mixup/CutMix**: Mix training samples
2. **Test-Time Augmentation (TTA)**: Average predictions over augmented versions
3. **Ensemble**: Train 3-5 models with different seeds, average predictions
4. **Class balancing**: Weighted random sampling in DataLoader
5. **Progressive resizing**: Start with smaller spectrograms, increase

### 5.4 Experiment Tracking

Use TensorBoard or Weights & Biases:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_01')

# Log metrics each epoch
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/murmur', murmur_acc, epoch)
writer.add_scalar('Accuracy/outcome', outcome_acc, epoch)
```

### 5.5 Model Selection

Save best model based on:
- **Primary**: Combined validation accuracy (avg of murmur + outcome)
- **Secondary**: Lowest validation loss

```python
if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config
    }, 'checkpoints/best_model.pt')
```

---

## Research Areas
- Learning rate schedulers (OneCycleLR vs CosineAnnealing)
- Label smoothing effect on generalization
- K-fold cross-validation for robust estimates

---

## Expected Outcome
- Trained model checkpoint with 95%+ accuracy
- Training logs and TensorBoard visualizations
- Hyperparameter tuning results documented
- Best model saved to `checkpoints/best_model.pt`

---

## Estimated Effort
- **1-3 days** depending on experimentation needed

---

## Dependencies
- Step 4: Training pipeline
