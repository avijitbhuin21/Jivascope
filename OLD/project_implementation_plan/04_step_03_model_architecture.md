# Step 3: Model Architecture Design

## Objective
Design and implement the multi-task CNN model for joint Murmur Detection and Clinical Outcome prediction.

## Prerequisites
- Step 2 completed (data pipeline ready)
- Understanding of spectrogram dimensions

---

## Implementation Details

### 3.1 Model Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT                                │
│           3-channel spectrogram (3, 128, 256)           │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 BACKBONE (Pretrained)                   │
│        EfficientNet-B0 or ResNet18 (ImageNet)           │
│              Output: (1280/512, H', W')                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              GLOBAL AVERAGE POOLING                     │
│                 Output: (1280/512)                      │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│   MURMUR HEAD       │       │   OUTCOME HEAD      │
│   FC(512) → ReLU    │       │   FC(512) → ReLU    │
│   Dropout(0.3)      │       │   Dropout(0.3)      │
│   FC(3) → Softmax   │       │   FC(1) → Sigmoid   │
│   (3-class)         │       │   (Binary)          │
└─────────────────────┘       └─────────────────────┘
```

### 3.2 Backbone Implementation

Create `src/models/backbone.py`:

```python
import torch
import torch.nn as nn
import torchvision.models as models

def create_backbone(name: str = 'efficientnet_b0', pretrained: bool = True):
    """
    Create pretrained backbone for feature extraction.
    
    Options:
    - 'efficientnet_b0': 1280-dim output, 5.3M params
    - 'resnet18': 512-dim output, 11.7M params
    - 'resnet34': 512-dim output, 21.8M params
    """
    if name == 'efficientnet_b0':
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove classifier head
        model.classifier = nn.Identity()
        feature_dim = 1280
    elif name == 'resnet18':
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        model.fc = nn.Identity()
        feature_dim = 512
    
    return model, feature_dim
```

### 3.3 Classification Heads

Create `src/models/classifier.py`:

```python
class ClassificationHead(nn.Module):
    """Single classification head with dropout regularization."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, 
                 dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 3.4 Full Model Assembly

Create `src/models/model.py`:

```python
class HeartSoundClassifier(nn.Module):
    """
    Multi-task model for heart sound classification.
    
    Outputs:
    - murmur_logits: (B, 3) for Absent/Present/Unknown
    - outcome_logits: (B, 1) for Normal/Abnormal
    """
    
    def __init__(self, backbone_name: str = 'efficientnet_b0', 
                 pretrained: bool = True,
                 hidden_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        self.backbone, feature_dim = create_backbone(backbone_name, pretrained)
        
        self.murmur_head = ClassificationHead(
            feature_dim, hidden_dim, num_classes=3, dropout=dropout
        )
        self.outcome_head = ClassificationHead(
            feature_dim, hidden_dim, num_classes=1, dropout=dropout
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        murmur_logits = self.murmur_head(features)
        outcome_logits = self.outcome_head(features)
        
        return {
            'murmur': murmur_logits,
            'outcome': outcome_logits
        }
```

### 3.5 Optional: Attention Mechanism

For improved focus on relevant regions, add CBAM (Convolutional Block Attention Module):

```python
# Research and implement if accuracy needs boost
class CBAM(nn.Module):
    """Channel and Spatial Attention Module."""
    pass
```

---

## Research Areas
- Compare EfficientNet-B0 vs ResNet18 performance
- Investigate deeper models (EfficientNet-B3, ResNet50) if accuracy insufficient
- Research attention mechanisms (SE-Net, CBAM) for audio spectrograms

---

## Expected Outcome
- `backbone.py` with pretrained backbone factory
- `classifier.py` with classification head module
- `model.py` with full multi-task model
- Model summary showing parameter count (~6M for EfficientNet-B0)

---

## Estimated Effort
- **4-6 hours** for model implementation and testing

---

## Dependencies
- Step 2: Data preprocessing (for spectrogram dimensions)
