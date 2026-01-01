# AST Heart Sound Classifier

Audio Spectrogram Transformer (AST) model for heart sound classification and murmur detection.

## Overview

This pipeline uses the pre-trained AST model from Hugging Face's `transformers` library, fine-tuned for heart sound classification.

**Input**: WAV audio file (any sample rate, will be converted to 16kHz)
**Output**: Binary predictions for `heart_sound_present` and `murmur_present`

## Usage

### Training

```powershell
# Auto-detect GPU/CPU and train
python train.py

# Specify profile
python train.py --profile cpu
python train.py --profile l4

# Custom parameters
python train.py --epochs 50 --batch-size 8 --lr 0.00005

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
  "inference_time_ms": 250.5,
  "metadata": {
    "model_type": "ast",
    "model_version": "ast_v1"
  }
}
```

## Architecture

- **Base Model**: MIT/ast-finetuned-audioset-10-10-0.4593
- **Input Features**: Log-mel spectrogram (128 mel bins, 16kHz)
- **Classification Head**: 768 → 256 → 64 → 2

## Requirements

```powershell
pip install -r requirements.txt
```

Key dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- librosa>=0.10.0
