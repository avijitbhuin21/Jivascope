"""
Heart Sound Predictor Service.

Loads LightCardiacNet model at startup and provides prediction API for heart sound analysis.
"""

import os
import io
import base64
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, hilbert


SAMPLE_RATE = 4000
TARGET_DURATION = 10.0
N_MFCC = 13
N_FFT = 256
HOP_LENGTH = 64

MODEL_PATH = Path(__file__).parent / "model" / "lightcardiacnet.pt"


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, rnn_output: torch.Tensor) -> tuple:
        weights = torch.nn.functional.softmax(self.attention(rnn_output), dim=1)
        context = torch.sum(weights * rnn_output, dim=1)
        return context, weights.squeeze(-1)


class BiGRUNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = 39,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = AttentionLayer(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        rnn_out, _ = self.bigru(x)
        context, attn_weights = self.attention(rnn_out)
        logits = self.classifier(context)
        return logits, attn_weights


class LightCardiacNet(nn.Module):
    def __init__(
        self,
        input_size: int = 39,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.network1 = BiGRUNetwork(input_size, hidden_size, num_layers, num_classes, dropout)
        self.network2 = BiGRUNetwork(input_size, hidden_size, num_layers, num_classes, dropout)
        
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> tuple:
        logits1, attn1 = self.network1(x)
        logits2, attn2 = self.network2(x)
        
        w = torch.sigmoid(self.ensemble_weight)
        ensemble_logits = w * logits1 + (1 - w) * logits2
        
        return ensemble_logits, (attn1, attn2)


class LightCardiacNetSingle(nn.Module):
    def __init__(
        self,
        input_size: int = 39,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.network = BiGRUNetwork(input_size, hidden_size, num_layers, num_classes, dropout)
    
    def forward(self, x: torch.Tensor) -> tuple:
        return self.network(x)


def create_model(
    model_type: str = 'ensemble',
    input_size: int = 39,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.3
) -> nn.Module:
    if model_type == 'ensemble':
        return LightCardiacNet(input_size, hidden_size, num_layers, num_classes, dropout)
    elif model_type == 'single':
        return LightCardiacNetSingle(input_size, hidden_size, num_layers, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio


def apply_bandpass_filter(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    low_freq: float = 25.0,
    high_freq: float = 400.0
) -> np.ndarray:
    nyquist = sr / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)
    
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)
    
    return filtered.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms
    return audio


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    current_length = len(audio)
    
    if current_length > target_length:
        start = (current_length - target_length) // 2
        return audio[start:start + target_length]
    elif current_length < target_length:
        pad_total = target_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0)
    
    return audio


def extract_features_from_array(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_duration: float = TARGET_DURATION,
    apply_filter: bool = True
) -> torch.Tensor:
    if apply_filter:
        audio = apply_bandpass_filter(audio, sr)
    
    audio = normalize_audio(audio)
    
    target_length = int(target_duration * sr)
    audio = pad_or_truncate(audio, target_length)
    
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.vstack([mfcc, delta, delta2])
    features = features.T
    
    return torch.tensor(features, dtype=torch.float32)


def generate_spectrogram_image(audio: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        fmax=sr // 2
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    img = librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=512,
        x_axis='time',
        y_axis='mel',
        cmap='viridis',
        ax=ax
    )
    
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(cbar.ax.get_yticklabels(), color='white')
    
    ax.set_title('Mel Spectrogram', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time', color='white')
    ax.set_ylabel('Hz', color='white')
    ax.tick_params(axis='both', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def estimate_bpm(audio: np.ndarray, sr: int) -> int:
    try:
        analytic_signal = hilbert(audio)
        envelope = np.abs(analytic_signal)
        
        envelope_smooth = np.convolve(envelope, np.ones(int(sr * 0.05)) / int(sr * 0.05), mode='same')
        
        min_distance = int(sr * 0.4)
        threshold = np.mean(envelope_smooth) + 0.5 * np.std(envelope_smooth)
        
        peaks, _ = find_peaks(envelope_smooth, distance=min_distance, height=threshold)
        
        if len(peaks) < 2:
            peaks, _ = find_peaks(envelope_smooth, distance=min_distance)
        
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / sr
            avg_interval = np.median(intervals)
            bpm = int(60 / avg_interval)
            
            if 40 <= bpm <= 200:
                return bpm
        
        return 72
        
    except Exception:
        return 72


class HeartSoundPredictor:
    """Predictor class for heart sound classification using LightCardiacNet."""
    
    _instance: Optional['HeartSoundPredictor'] = None
    _model_loaded: bool = False
    
    def __init__(self):
        self.device = 'cpu'
        self.model = None
        
    @classmethod
    def get_instance(cls) -> 'HeartSoundPredictor':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_model(self) -> None:
        if self._model_loaded:
            print("Model already loaded.")
            return
            
        print(f"Loading LightCardiacNet model from {MODEL_PATH}...")
        
        start_time = time.time()
        
        checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)
        
        config = checkpoint.get('config', {})
        self.model = create_model(
            model_type=config.get('model_type', 'ensemble'),
            input_size=config.get('input_size', 39),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        load_time = time.time() - start_time
        print(f"LightCardiacNet loaded successfully in {load_time:.2f} seconds on {self.device}")
        
        HeartSoundPredictor._model_loaded = True
    
    def preprocess_audio(self, audio_path: str) -> tuple[torch.Tensor, np.ndarray, int]:
        audio = load_audio(audio_path, target_sr=SAMPLE_RATE)
        audio = apply_bandpass_filter(audio, sr=SAMPLE_RATE)
        audio = normalize_audio(audio)
        
        target_length = int(TARGET_DURATION * SAMPLE_RATE)
        audio_processed = pad_or_truncate(audio, target_length)
        
        features = extract_features_from_array(audio_processed, sr=SAMPLE_RATE, apply_filter=False)
        
        return features.unsqueeze(0), audio_processed, SAMPLE_RATE
    
    @torch.no_grad()
    def predict(self, audio_bytes: bytes) -> dict:
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            features, audio_for_spec, sr = self.preprocess_audio(tmp_path)
            features = features.to(self.device)
            
            spectrogram_image = generate_spectrogram_image(audio_for_spec, sr)
            
            logits, (attn1, attn2) = self.model(features)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
            inference_time = (time.time() - start_time) * 1000
            
            heart_sound_present = bool(probs[0] > 0.5)
            murmur_present = bool(probs[1] > 0.5) if heart_sound_present else False
            
            result = {
                "success": True,
                "prediction": {
                    "heart_sound_present": heart_sound_present,
                    "murmur_present": murmur_present
                },
                "confidence": {
                    "heart_sound": float(probs[0]),
                    "murmur": float(probs[1])
                },
                "spectrogram": spectrogram_image,
                "bpm": estimate_bpm(audio_for_spec, sr),
                "inference_time_ms": round(inference_time, 2)
            }
            
            return result
            
        finally:
            os.unlink(tmp_path)


predictor = HeartSoundPredictor.get_instance()
