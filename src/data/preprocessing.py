"""
Audio Preprocessing Pipeline for Heart Sound Classification
Handles audio loading, normalization, and spectrogram generation
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional


SAMPLE_RATE = 4000
TARGET_DURATION = 10.0
N_MELS = 128
N_FFT = 256
HOP_LENGTH = 64


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: 4000 Hz)
    
    Returns:
        Audio signal as numpy array
    """
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Z-score normalization of audio signal.
    
    Args:
        audio: Raw audio signal
    
    Returns:
        Normalized audio signal with zero mean and unit variance
    """
    if audio.std() == 0:
        return audio
    return (audio - audio.mean()) / audio.std()


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Ensure audio has consistent length by padding or truncating.
    
    Args:
        audio: Audio signal
        target_length: Target number of samples
    
    Returns:
        Audio signal with exactly target_length samples
    """
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


def apply_bandpass_filter(
    audio: np.ndarray, 
    sr: int = SAMPLE_RATE, 
    low_freq: float = 25.0, 
    high_freq: float = 400.0
) -> np.ndarray:
    """
    Apply bandpass filter to remove noise outside heart sound frequency range.
    Heart sounds typically range from 25-400 Hz.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        low_freq: Low cutoff frequency
        high_freq: High cutoff frequency
    
    Returns:
        Filtered audio signal
    """
    from scipy.signal import butter, filtfilt
    
    nyquist = sr / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)
    
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)
    
    return filtered.astype(np.float32)


def create_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Generate Mel spectrogram from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
    
    Returns:
        Mel spectrogram in dB scale, shape (n_mels, time_frames)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def create_mfcc(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 40,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """
    Generate MFCC features from audio.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length between frames
    
    Returns:
        MFCC features, shape (n_mfcc, time_frames)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc


def create_delta_features(features: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Compute delta (derivative) features.
    
    Args:
        features: Input features (e.g., mel spectrogram)
        order: Order of derivative (1 for delta, 2 for delta-delta)
    
    Returns:
        Delta features
    """
    return librosa.feature.delta(features, order=order)


def create_multichannel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    target_height: int = 128,
    target_width: int = 626
) -> np.ndarray:
    """
    Create a 3-channel spectrogram for CNN input.
    
    Channels:
        - Channel 0 (R): Mel Spectrogram
        - Channel 1 (G): Delta Mel (temporal dynamics)
        - Channel 2 (B): Delta-Delta Mel (acceleration)
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length
        target_height: Target spectrogram height
        target_width: Target spectrogram width
    
    Returns:
        3-channel spectrogram, shape (3, target_height, target_width)
    """
    mel_spec = create_mel_spectrogram(audio, sr, n_mels, n_fft, hop_length)
    delta = create_delta_features(mel_spec, order=1)
    delta2 = create_delta_features(mel_spec, order=2)
    
    mel_spec = normalize_spectrogram(mel_spec)
    delta = normalize_spectrogram(delta)
    delta2 = normalize_spectrogram(delta2)
    
    mel_spec = resize_spectrogram(mel_spec, target_height, target_width)
    delta = resize_spectrogram(delta, target_height, target_width)
    delta2 = resize_spectrogram(delta2, target_height, target_width)
    
    multichannel = np.stack([mel_spec, delta, delta2], axis=0)
    
    return multichannel.astype(np.float32)


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram to [0, 1] range.
    
    Args:
        spec: Input spectrogram
    
    Returns:
        Normalized spectrogram
    """
    spec_min = spec.min()
    spec_max = spec.max()
    
    if spec_max - spec_min == 0:
        return np.zeros_like(spec)
    
    return (spec - spec_min) / (spec_max - spec_min)


def resize_spectrogram(spec: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Resize spectrogram to target dimensions using interpolation.
    
    Args:
        spec: Input spectrogram (height, width)
        target_height: Target height
        target_width: Target width
    
    Returns:
        Resized spectrogram
    """
    from scipy.ndimage import zoom
    
    current_height, current_width = spec.shape
    zoom_h = target_height / current_height
    zoom_w = target_width / current_width
    
    resized = zoom(spec, (zoom_h, zoom_w), order=1)
    
    return resized


def preprocess_audio_file(
    file_path: str,
    target_duration: float = TARGET_DURATION,
    sr: int = SAMPLE_RATE,
    apply_filter: bool = True,
    n_mels: int = N_MELS
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single audio file.
    
    Args:
        file_path: Path to audio file
        target_duration: Target duration in seconds
        sr: Sample rate
        apply_filter: Whether to apply bandpass filter
        n_mels: Number of mel bands
    
    Returns:
        3-channel spectrogram ready for model input
    """
    audio = load_audio(file_path, sr)
    
    if apply_filter:
        audio = apply_bandpass_filter(audio, sr)
    
    audio = normalize_audio(audio)
    
    target_length = int(target_duration * sr)
    audio = pad_or_truncate(audio, target_length)
    
    spectrogram = create_multichannel_spectrogram(audio, sr, n_mels)
    
    return spectrogram


if __name__ == "__main__":
    print("Preprocessing module loaded successfully")
    print(f"Default settings: SR={SAMPLE_RATE}, Duration={TARGET_DURATION}s, Mels={N_MELS}")
