# Step 5: Model Pruning & Optimization

## Objective
Apply static network pruning to achieve lightweight model for CPU deployment.

## Tasks

### 5.1 Baseline Model Profiling
Before pruning:
```python
from thop import profile

model = LightCardiacNet()
input = torch.randn(1, 625, 39)  # (batch, seq_len, features)
macs, params = profile(model, inputs=(input,))
print(f"MACs: {macs}, Params: {params}")
```

### 5.2 Apply Structured Pruning
```python
import torch_pruning as tp

# Create pruner
imp = tp.importance.MagnitudeImportance(p=2)
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs=torch.randn(1, 625, 39),
    importance=imp,
    pruning_ratio=0.5,  # Prune 50% of channels
    ignored_layers=[]
)

# Prune
pruner.step()

# Fine-tune after pruning (5-10 epochs)
fine_tune(model, train_loader, epochs=10)
```

### 5.3 Quantization (Optional)
```python
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {nn.GRU, nn.Linear},
    dtype=torch.qint8
)
```

### 5.4 Benchmark Inference Speed
```python
import time

def benchmark_inference(model, n_samples=100):
    model.eval()
    input = torch.randn(1, 625, 39)
    
    times = []
    for _ in range(n_samples):
        start = time.time()
        with torch.no_grad():
            _ = model(input)
        times.append(time.time() - start)
    
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'max_ms': np.max(times) * 1000
    }
```

### 5.5 Save Optimized Model
```python
# Save pruned model
torch.save(model.state_dict(), 'lightcardiacnet_pruned.pt')

# Save for inference (TorchScript)
scripted = torch.jit.script(model)
scripted.save('lightcardiacnet_optimized.pt')
```

## Deliverables
- [ ] Pruned model with 50%+ reduction
- [ ] Inference benchmark showing <100ms
- [ ] `model/pruning.py` - Pruning utilities

## Estimated Time
~2-3 hours
