# Requirements / Tech Stack

## Core Dependencies

```
# Deep Learning Framework
torch>=2.0.0
torchaudio>=2.0.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.1
scipy>=1.10.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0

# Visualization & Monitoring
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0

# Model Optimization
torch-pruning>=1.2.0

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
```

## Python Version
- Python 3.10+ recommended

## Hardware Requirements
- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, 8-core CPU
- **GPU**: Optional (model optimized for CPU)

## Installation

```bash
cd NEW
pip install -r planning/01_requirements.txt
```

## Key Libraries Purpose

| Library | Purpose |
|---------|---------|
| `torch` | Neural network framework |
| `librosa` | Audio loading, MFCC extraction |
| `torch-pruning` | Network pruning for lightweight model |
| `pandas` | CSV/data handling |
| `tqdm` | Progress bars for training |
