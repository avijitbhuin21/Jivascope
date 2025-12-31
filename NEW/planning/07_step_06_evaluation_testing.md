# Step 6: Evaluation & Testing

## Objective
Comprehensive evaluation on test set with detailed metrics.

## Tasks

### 6.1 Test Set Evaluation
```python
def evaluate(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            logits, _ = model(features)
            probs = torch.sigmoid(logits)
            all_preds.append(probs)
            all_labels.append(labels)
    
    return compute_metrics(all_preds, all_labels)
```

### 6.2 Metrics to Report
```python
metrics = {
    'heart_sound': {
        'accuracy': ...,
        'precision': ...,
        'recall': ...,
        'f1': ...
    },
    'murmur': {
        'accuracy': ...,
        'sensitivity': ...,
        'specificity': ...,
        'f1': ...,
        'auc_roc': ...
    },
    'overall': {
        'weighted_accuracy': ...,
        'macro_f1': ...
    }
}
```

### 6.3 Confusion Matrix
Generate per-class confusion matrices:
- Heart Sound: Normal vs Abnormal
- Murmur: Absent vs Present

### 6.4 Attention Visualization
```python
def visualize_attention(audio_path, model):
    features = extract_features(audio_path)
    _, (attn1, attn2) = model(features.unsqueeze(0))
    
    plt.figure(figsize=(12, 4))
    plt.plot(attn1.squeeze().numpy())
    plt.title('Attention Weights Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.savefig('attention_visualization.png')
```

### 6.5 Error Analysis
- Identify misclassified samples
- Analyze patterns in errors
- Check for data quality issues

## Deliverables
- [ ] Test set classification report
- [ ] Confusion matrices
- [ ] Attention visualizations
- [ ] `evaluate.py` - Evaluation script

## Estimated Time
~2-3 hours
