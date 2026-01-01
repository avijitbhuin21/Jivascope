# PANNs Heart Sound Classifier

Pre-trained Audio Neural Networks (PANNs) Cnn14 model for heart sound classification and murmur detection.

## Overview

This pipeline uses the Cnn14 architecture from PANNs, optionally with pre-trained AudioSet weights, fine-tuned for heart sound classification.

**Input**: WAV audio file (any sample rate, will be converted to 32kHz)
**Output**: Binary predictions for `heart_sound_present` and `murmur_present`

## Usage

### Training

```powershell
# Auto-detect GPU/CPU and train
python train.py

# Specify profile
python train.py --profile cpu
python train.py --profile l4

# Use lighter model (faster, less memory)
python train.py --model-type light

# Custom parameters
python train.py --epochs 50 --batch-size 8 --lr 0.0001

# With pretrained weights
python train.py --pretrained path/to/Cnn14_mAP=0.431.pth

# Freeze encoder (only train classifier head)
python train.py --freeze-encoder
```

### Prediction

```powershell
python predict.py --audio path/to/audio.wav
python predict.py --audio path/to/audio.wav --model model/checkpoints/best_model.pt
```

### Output Format

```json
{
  "success": true,
  "prediction": {
    "heart_sound_present": true,
    "murmur_present": false
  },
  "confidence": {
    "heart_sound": 0.97,
    "murmur": 0.12
  },
  "inference_time_ms": 150.2,
  "metadata": {
    "model_type": "panns",
    "model_version": "panns_v1"
  }
}
```

## Architecture

- **Cnn14**: Full PANNs architecture with 80M+ parameters
- **Light**: Lighter version with ~5M parameters (faster training)
- **Input Features**: Log-mel spectrogram (64 mel bins, 32kHz)
- **Classification Head**: 2048 → 512 → 128 → 2

## Pre-trained Weights (Optional)

Download Cnn14 weights from: https://zenodo.org/record/3987831

```powershell
python train.py --pretrained Cnn14_mAP=0.431.pth
```

## Requirements

```powershell
pip install -r requirements.txt
```

Key dependencies:
- torch>=2.0.0
- librosa>=0.10.0
