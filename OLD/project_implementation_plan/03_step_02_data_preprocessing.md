# Step 2: Data Preprocessing & Dataset Creation

## Objective
Build a robust data pipeline that converts raw audio files into model-ready features (spectrograms) and creates PyTorch Dataset/DataLoader for training.

## Prerequisites
- Step 1 completed (environment setup, EDA done)
- Understanding of class distribution and audio characteristics

---

## Implementation Details

### 2.1 Audio Preprocessing Pipeline

Create `src/data/preprocessing.py`:

```python
# Key functions to implement:

def load_audio(file_path: str, target_sr: int = 4000) -> np.ndarray:
    """Load and resample audio file."""
    pass

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Z-score normalization of audio signal."""
    pass

def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Ensure all audio samples have consistent length."""
    pass

def create_mel_spectrogram(audio: np.ndarray, sr: int, n_mels: int = 128, 
                           n_fft: int = 512, hop_length: int = 128) -> np.ndarray:
    """Generate Mel spectrogram from audio."""
    pass

def create_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """Generate MFCC features from audio."""
    pass
```

### 2.2 Multi-Channel Spectrogram Generation

Create 3-channel spectrogram images (like RGB) for pretrained CNN:

| Channel | Feature | Purpose |
|---------|---------|---------|
| Red | Mel Spectrogram | Frequency content |
| Green | Delta Mel | Temporal dynamics |
| Blue | Delta-Delta Mel | Acceleration features |

Alternatively:
| Channel | Feature |
|---------|---------|
| Red | Mel Spectrogram |
| Green | MFCC |
| Blue | Chromagram |

### 2.3 Train/Validation/Test Split

Create `src/data/split.py`:

**Important**: Split by **PATIENT ID**, not by recording, to prevent data leakage.

```python
def create_patient_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: str = 'Outcome'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by patient ID with stratification.
    Ensures all recordings from same patient are in same split.
    """
    pass
```

**Target split**:
- Train: 70% of patients
- Validation: 15% of patients
- Test: 15% of patients (held out until final evaluation)

### 2.4 PyTorch Dataset Class

Create `src/data/dataset.py`:

```python
class HeartSoundDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for heart sound classification.
    
    Each sample returns:
    - spectrogram: (3, H, W) tensor
    - murmur_label: int (0=Absent, 1=Present, 2=Unknown)
    - outcome_label: int (0=Normal, 1=Abnormal)
    - patient_id: str (for analysis)
    """
    
    def __init__(self, df, audio_dir, transform=None, 
                 target_duration=5.0, sample_rate=4000):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
```

### 2.5 Data Augmentation

Create `src/data/augmentation.py`:

**Training-time augmentations**:
1. Time shift (random shift within bounds)
2. Pitch shifting (±2 semitones)
3. Noise injection (Gaussian noise, SNR 20-40dB)
4. Time stretching (0.8x - 1.2x)
5. SpecAugment (frequency/time masking on spectrograms)

**Validation/Test**: No augmentation

---

## Research Areas
- Optimal spectrogram parameters for 4000 Hz audio
- SpecAugment implementation details
- Handling variable-length recordings (padding vs truncation vs segmentation)

---

## Expected Outcome
- `preprocessing.py` with all audio processing functions
- `dataset.py` with PyTorch Dataset class
- `augmentation.py` with audio augmentation transforms
- `split.py` with patient-level stratified splitting
- Pre-computed spectrograms saved to disk (optional, for faster training)

---

## Estimated Effort
- **1-2 days** for complete data pipeline implementation

---

## Dependencies
- Step 1: Environment Setup
