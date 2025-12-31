# Jivascope - Heart Murmur Detection System

## Project Overview

An AI-powered heart sound classification system using **LightCardiacNet** architecture - a lightweight, attention-based Bi-GRU ensemble network designed for real-time cardiac sound analysis.

---

## Architecture: LightCardiacNet (Bi-GRU)

### Model Summary
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LIGHTCARDIACNET ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AUDIO INPUT (.wav) ──► MFCC Features ──► [Bi-GRU + Attention] ──► OUTPUT   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        ENSEMBLE STRUCTURE                             │   │
│  │                                                                       │   │
│  │   ┌─────────────────────┐         ┌─────────────────────┐            │   │
│  │   │  Bi-GRU Network 1   │         │  Bi-GRU Network 2   │            │   │
│  │   │  + Attention Layer  │         │  + Attention Layer  │            │   │
│  │   │  (Pruned/Sparse)    │         │  (Pruned/Sparse)    │            │   │
│  │   └──────────┬──────────┘         └──────────┬──────────┘            │   │
│  │              │                               │                        │   │
│  │              └───────────┬───────────────────┘                        │   │
│  │                          │                                            │   │
│  │                 [Weighted Average Fusion]                             │   │
│  │                          │                                            │   │
│  │                          ▼                                            │   │
│  │              ┌───────────────────────┐                                │   │
│  │              │   Final Prediction    │                                │   │
│  │              │ heart_sound: bool     │                                │   │
│  │              │ murmur: bool          │                                │   │
│  │              └───────────────────────┘                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Performance: 98.5% accuracy | 18ms inference | Lightweight (~sparse)       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components
1. **Bi-GRU Networks**: Bidirectional GRU layers capture temporal patterns in heart sounds
2. **Attention Mechanism**: Extracts salient features, reduces computational complexity
3. **Network Pruning**: Static pruning makes model lightweight for CPU deployment
4. **Ensemble Fusion**: Weighted average of two networks for robust predictions

---

## Expected Input/Output

### Input
```json
{
  "type": "audio_file",
  "format": "WAV",
  "sample_rate": 4000,
  "duration": "variable (will be padded/truncated to 10s)",
  "channels": "mono",
  "example": "patient_recording.wav"
}
```

### Output
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
  "inference_time_ms": 18,
  "metadata": {
    "model_version": "lightcardiacnet_v1",
    "processing_date": "2024-12-31T18:00:00Z"
  }
}
```

### Prediction Logic
```
IF heart_sound_present == false:
    murmur_present = false  (cannot detect murmur without heart sound)
ELSE:
    murmur_present = model_prediction
```

---

## Multi-Valve Processing (For CirCor Dataset)

When multiple valve recordings exist per patient:

```
Patient Input: [AV.wav, MV.wav, PV.wav, TV.wav]
                    │        │        │        │
                    ▼        ▼        ▼        ▼
                 [Model]  [Model]  [Model]  [Model]
                    │        │        │        │
                    └────────┴────────┴────────┘
                               │
                    [Decision-Level Fusion]
                    (Max confidence voting)
                               │
                               ▼
                    Final Patient Prediction
```

---

## Target Specifications

| Metric | Target | Notes |
|--------|--------|-------|
| Overall Accuracy | ≥95% | LightCardiacNet reports 98.5% |
| Inference Time | <6 seconds | LightCardiacNet: 18ms per file |
| Platform | CPU only | No GPU required |
| Model Size | <10MB | Achieved via pruning |
| Output Format | JSON | Structured response |

---

## Data Pipeline

```
Raw Audio (.wav)
      │
      ▼
┌─────────────┐
│ Load Audio  │  librosa, 4kHz sample rate
└─────────────┘
      │
      ▼
┌─────────────┐
│ Bandpass    │  25-400 Hz (heart sound range)
│ Filter      │
└─────────────┘
      │
      ▼
┌─────────────┐
│ RMS         │  Normalize amplitude
│ Normalize   │
└─────────────┘
      │
      ▼
┌─────────────┐
│ Pad/Truncate│  Target: 10 seconds
└─────────────┘
      │
      ▼
┌─────────────┐
│ MFCC        │  13 coefficients + deltas
│ Extraction  │
└─────────────┘
      │
      ▼
Model Input (features)
```

---

## Project Structure

```
NEW/
├── model/                    # Model implementation
│   ├── __init__.py
│   ├── lightcardiacnet.py    # Main model architecture
│   ├── attention.py          # Attention layer
│   ├── dataset.py            # Data loading
│   └── utils.py              # Helper functions
│
├── planning/                 # Documentation
│   ├── 00_overview.md        # This file
│   ├── 01_requirements.txt   # Dependencies
│   ├── 02_step_01_*.md       # Implementation steps
│   └── implementation.md     # Progress tracking
│
├── cleaned_data/             # Organized audio files
│   ├── train/
│   ├── test/
│   └── val/
│
├── cleaned_data_entries/     # CSV metadata
│   ├── train.csv
│   ├── test.csv
│   ├── val.csv
│   └── cleaned_data.csv
│
└── src/                      # Existing feature extraction
    └── features/
        └── multi_channel_features.py
```
