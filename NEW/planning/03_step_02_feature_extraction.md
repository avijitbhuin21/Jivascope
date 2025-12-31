# Step 2: Feature Extraction Pipeline

## Objective
Implement MFCC-based feature extraction for LightCardiacNet.

## Tasks

### 2.1 Reuse Existing Feature Extraction
Leverage `src/features/multi_channel_features.py`:
- MFCC extraction with deltas ✓
- Bandpass filtering (25-400 Hz) ✓
- RMS normalization ✓
- Pad/truncate to 10 seconds ✓

### 2.2 Adapt for LightCardiacNet
Modify for Bi-GRU input format:
```python
def extract_features_for_bigru(audio_path):
    """
    Extract MFCC features formatted for Bi-GRU input.
    
    Returns:
        features: (seq_len, n_features) tensor
    """
    audio = load_and_preprocess(audio_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=4000, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.vstack([mfcc, delta, delta2])  # (39, time_steps)
    return features.T  # (time_steps, 39) for RNN input
```

### 2.3 Add Data Augmentation
```python
augmentations = [
    TimeShift(max_shift=0.1),
    SpeedPerturbation(range=(0.9, 1.1)),
    NoiseInjection(snr_range=(15, 30)),
    VolumeScaling(range=(0.9, 1.1))
]
```

### 2.4 Create Feature Cache
Pre-compute features to speed up training:
```
NEW/cached_features/
├── train/
├── val/
└── test/
```

## Deliverables
- [ ] `model/features.py` - Feature extraction module
- [ ] Cached features for all splits
- [ ] Augmentation pipeline

## Estimated Time
~2-3 hours
