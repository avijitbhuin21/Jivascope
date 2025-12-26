# Step 6: Inference Pipeline & Evaluation

## Objective
Create a clean inference pipeline that accepts audio-only input (no TSV required) and provides predictions for Murmur and Outcome.

## Prerequisites
- Step 5 completed (trained model checkpoint available)

---

## Implementation Details

### 6.1 Audio-Only Preprocessing

Create `src/inference/audio_processor.py`:

```python
import torch
import torchaudio
import librosa
import numpy as np

class AudioProcessor:
    """
    Preprocesses raw audio for inference.
    No TSV/segmentation required.
    """
    
    def __init__(self, 
                 sample_rate: int = 4000,
                 target_duration: float = 5.0,
                 n_mels: int = 128,
                 n_fft: int = 512,
                 hop_length: int = 128):
        self.sample_rate = sample_rate
        self.target_length = int(target_duration * sample_rate)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and resample if needed."""
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """Full preprocessing pipeline."""
        # Normalize
        audio = (audio - audio.mean()) / (audio.std() + 1e-8)
        
        # Pad or truncate to target length
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]
        
        # Create multi-channel spectrogram
        spectrogram = self._create_spectrogram(audio)
        
        return torch.from_numpy(spectrogram).float()
    
    def _create_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Create 3-channel spectrogram."""
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, 
            n_mels=self.n_mels, n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Delta and delta-delta
        delta = librosa.feature.delta(mel_db)
        delta2 = librosa.feature.delta(mel_db, order=2)
        
        # Normalize each channel
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        delta = (delta - delta.mean()) / (delta.std() + 1e-8)
        delta2 = (delta2 - delta2.mean()) / (delta2.std() + 1e-8)
        
        # Stack into 3-channel image
        spectrogram = np.stack([mel_db, delta, delta2], axis=0)
        
        return spectrogram
    
    def process_file(self, file_path: str) -> torch.Tensor:
        """Complete pipeline from file path to tensor."""
        audio = self.load_audio(file_path)
        return self.preprocess(audio)
```

### 6.2 Predictor Class

Create `src/inference/predictor.py`:

```python
import torch
from typing import Dict, Union
from pathlib import Path

class HeartSoundPredictor:
    """
    Inference wrapper for heart sound classification.
    
    Usage:
        predictor = HeartSoundPredictor('checkpoints/best_model.pt')
        result = predictor.predict('path/to/audio.wav')
        print(result['murmur'])  # 'Present', 'Absent', or 'Unknown'
        print(result['outcome'])  # 'Normal' or 'Abnormal'
    """
    
    MURMUR_CLASSES = ['Absent', 'Present', 'Unknown']
    OUTCOME_CLASSES = ['Normal', 'Abnormal']
    
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Recreate model architecture
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(self.device)
        
        # Audio processor
        self.processor = AudioProcessor(
            sample_rate=self.config['data']['sample_rate'],
            target_duration=self.config['data']['target_duration'],
            n_mels=self.config['data']['n_mels'],
            n_fft=self.config['data']['n_fft'],
            hop_length=self.config['data']['hop_length']
        )
    
    def _build_model(self):
        """Reconstruct model from config."""
        from models.model import HeartSoundClassifier
        return HeartSoundClassifier(
            backbone_name=self.config['model']['backbone'],
            pretrained=False,  # Not needed for inference
            hidden_dim=self.config['model']['hidden_dim'],
            dropout=self.config['model']['dropout']
        )
    
    @torch.no_grad()
    def predict(self, audio_path: Union[str, Path]) -> Dict:
        """
        Predict murmur presence and clinical outcome from audio file.
        
        Args:
            audio_path: Path to .wav audio file
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Preprocess
        spectrogram = self.processor.process_file(str(audio_path))
        spectrogram = spectrogram.unsqueeze(0).to(self.device)
        
        # Inference
        outputs = self.model(spectrogram)
        
        # Process murmur prediction
        murmur_probs = torch.softmax(outputs['murmur'], dim=1)[0]
        murmur_idx = murmur_probs.argmax().item()
        murmur_label = self.MURMUR_CLASSES[murmur_idx]
        murmur_confidence = murmur_probs[murmur_idx].item()
        
        # Process outcome prediction
        outcome_prob = torch.sigmoid(outputs['outcome'])[0, 0].item()
        outcome_label = 'Abnormal' if outcome_prob > 0.5 else 'Normal'
        outcome_confidence = outcome_prob if outcome_prob > 0.5 else 1 - outcome_prob
        
        return {
            'murmur': murmur_label,
            'murmur_confidence': round(murmur_confidence * 100, 2),
            'murmur_probabilities': {
                cls: round(prob.item() * 100, 2) 
                for cls, prob in zip(self.MURMUR_CLASSES, murmur_probs)
            },
            'outcome': outcome_label,
            'outcome_confidence': round(outcome_confidence * 100, 2),
            'outcome_probability_abnormal': round(outcome_prob * 100, 2)
        }
    
    def predict_batch(self, audio_paths: list) -> list:
        """Predict for multiple audio files."""
        return [self.predict(path) for path in audio_paths]
```

### 6.3 Final Evaluation on Test Set

Create `src/evaluation/evaluate.py`:

```python
def evaluate_on_test_set(predictor, test_df, audio_dir):
    """
    Evaluate model on held-out test set.
    
    Reports:
    - Overall accuracy
    - Per-class precision, recall, F1
    - Confusion matrix
    - ROC-AUC (for outcome)
    """
    pass
```

### 6.4 Demo Notebook

Create `notebooks/03_inference_demo.ipynb`:

```python
from src.inference.predictor import HeartSoundPredictor

# Load model
predictor = HeartSoundPredictor('checkpoints/best_model.pt')

# Single prediction
result = predictor.predict('path/to/test_audio.wav')
print(f"Murmur: {result['murmur']} ({result['murmur_confidence']}%)")
print(f"Outcome: {result['outcome']} ({result['outcome_confidence']}%)")
```

---

## Research Areas
- Threshold calibration for optimal sensitivity/specificity
- Uncertainty quantification (MC Dropout)
- Batch inference optimization

---

## Expected Outcome
- `audio_processor.py` with audio-only preprocessing
- `predictor.py` with simple inference API
- `evaluate.py` for test set evaluation
- `03_inference_demo.ipynb` showing usage
- Test set evaluation report with accuracy metrics

---

## Estimated Effort
- **4-6 hours** for inference pipeline
- **2-3 hours** for evaluation and documentation

---

## Dependencies
- Step 5: Trained model checkpoint
