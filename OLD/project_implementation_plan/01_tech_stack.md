# Technology Stack: Heart Sound Classification

## Recommended Architecture

Based on research, the highest accuracy approach for the CirCor dataset uses:
- **Combined Time-Frequency Representations** → ~99% accuracy
- **CNN-based architectures** (ResNet, EfficientNet)
- **Multi-task learning** for joint Murmur + Outcome prediction

---

## Core Technologies

### Python Version
- **Python 3.10+** (Colab Pro compatible)

### Deep Learning Framework
| Option | Recommendation | Reasoning |
|--------|----------------|-----------|
| **PyTorch** | ✅ Recommended | Flexible, excellent for research, strong community, TorchAudio support |
| TensorFlow | Alternative | Slightly easier deployment, but less flexible |

**Decision**: PyTorch with TorchAudio for audio processing

### Audio Processing
| Library | Purpose |
|---------|---------|
| **librosa** | Audio loading, feature extraction, augmentation |
| **torchaudio** | PyTorch-native audio processing, spectrograms |
| **scipy** | Signal processing utilities |

### Feature Representation
**Recommended: Multi-spectrogram approach** (highest accuracy on CirCor)

1. **Mel Spectrogram** - Captures frequency content in perceptual scale
2. **MFCC** - Mel-frequency cepstral coefficients for timbral features
3. **Scalogram (CWT)** - Wavelet-based time-frequency representation

### Model Architecture
| Component | Choice | Reasoning |
|-----------|--------|-----------|
| **Backbone** | EfficientNet-B0 or ResNet18 | Pretrained on ImageNet, transfer learning |
| **Input** | Multi-channel spectrogram (3 channels) | RGB-like input for pretrained models |
| **Heads** | Dual classification heads | Multi-task: Murmur (3-class) + Outcome (binary) |
| **Pooling** | Global Average Pooling | Reduce spatial dimensions |
| **Attention** | Optional CBAM or SE blocks | Focus on relevant time-frequency regions |

---

## Full Dependency List

```
# requirements.txt
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
wfdb>=4.1.0          # For reading .hea files
pywavelets>=1.4.0    # For wavelet transforms
audiomentations>=0.30.0  # Audio augmentation
tensorboard>=2.13.0  # Training visualization
```

---

## Project Structure

```
Jivascope/
├── project_implementation_plan/    # This planning folder
├── src/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   ├── preprocessing.py        # Audio preprocessing
│   │   ├── augmentation.py         # Data augmentation
│   │   └── split.py                # Train/val/test split
│   ├── models/
│   │   ├── backbone.py             # CNN backbone (EfficientNet/ResNet)
│   │   ├── classifier.py           # Multi-task classification heads
│   │   └── model.py                # Full model assembly
│   ├── training/
│   │   ├── trainer.py              # Training loop
│   │   ├── losses.py               # Custom loss functions
│   │   └── metrics.py              # Evaluation metrics
│   ├── inference/
│   │   ├── predictor.py            # Inference pipeline
│   │   └── audio_processor.py      # Audio-only preprocessing
│   └── utils/
│       ├── config.py               # Configuration management
│       └── visualization.py        # Plotting utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA notebook
│   ├── 02_training.ipynb           # Colab training notebook
│   └── 03_inference_demo.ipynb     # Inference demo
├── configs/
│   └── default.yaml                # Hyperparameters
├── checkpoints/                    # Saved models
└── tests/
    ├── test_dataset.py
    └── test_inference.py
```

---

## Hardware Requirements

| Environment | Specification |
|-------------|---------------|
| **Training** | Google Colab Pro (T4/V100 GPU, 16GB+ VRAM) |
| **Inference** | CPU-capable (laptop/server) |
| **RAM** | 8GB+ for data loading |
| **Storage** | ~2GB for dataset + ~500MB for checkpoints |

---

## Research Areas for Development Team

1. **Audio Augmentation**: Research optimal augmentation strategies (time stretch, pitch shift, noise injection)
2. **Class Imbalance**: The dataset has imbalanced classes - research focal loss, weighted sampling
3. **Ensemble Methods**: Research combining multiple models for higher accuracy
4. **Segmentation-Free Classification**: Deep dive into papers showing CNN can work without explicit S1/S2 segmentation
