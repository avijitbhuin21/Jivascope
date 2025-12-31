"""
Feature Extraction Module for LightCardiacNet

Extracts MFCC features with deltas for Bi-GRU input.
Output format: (time_steps, 39) tensor
"""

import numpy as np
import librosa
import torch
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional
import random


SAMPLE_RATE = 4000
TARGET_DURATION = 10.0
N_MFCC = 13
N_FFT = 256
HOP_LENGTH = 64


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


def extract_features_for_bigru(
    audio_path: str,
    sr: int = SAMPLE_RATE,
    target_duration: float = TARGET_DURATION,
    apply_filter: bool = True
) -> torch.Tensor:
    audio = load_audio(audio_path, sr)
    
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


class AudioAugmentation:
    def __init__(
        self,
        time_shift_max: float = 0.1,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        noise_snr_range: Tuple[float, float] = (15, 30),
        volume_range: Tuple[float, float] = (0.9, 1.1),
        augment_prob: float = 0.5
    ):
        self.time_shift_max = time_shift_max
        self.speed_range = speed_range
        self.noise_snr_range = noise_snr_range
        self.volume_range = volume_range
        self.augment_prob = augment_prob
    
    def time_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.augment_prob:
            return audio
        max_shift = int(len(audio) * self.time_shift_max)
        shift = random.randint(-max_shift, max_shift)
        return np.roll(audio, shift)
    
    def speed_perturbation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if random.random() > self.augment_prob:
            return audio
        speed = random.uniform(*self.speed_range)
        audio_stretched = librosa.effects.time_stretch(audio, rate=speed)
        return audio_stretched
    
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        if random.random() > self.augment_prob:
            return audio
        snr_db = random.uniform(*self.noise_snr_range)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise.astype(np.float32)
    
    def volume_scaling(self, audio: np.ndarray) -> np.ndarray:
        if random.random() > self.augment_prob:
            return audio
        scale = random.uniform(*self.volume_range)
        return audio * scale
    
    def __call__(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        audio = self.time_shift(audio, sr)
        audio = self.add_noise(audio)
        audio = self.volume_scaling(audio)
        return audio


if __name__ == "__main__":
    print(f"Feature extraction settings: SR={SAMPLE_RATE}, Duration={TARGET_DURATION}s, MFCC={N_MFCC}")
    print(f"Expected output shape: ({int(TARGET_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1}, {N_MFCC * 3})")
