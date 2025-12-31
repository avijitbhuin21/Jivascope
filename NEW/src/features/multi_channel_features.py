"""
Multi-Channel Feature Extraction for Heart Sound Classification

Features:
- Mel Spectrogram (base representation)
- MFCC with deltas (cepstral features)  
- Continuous Wavelet Transform (time-frequency localization)

Combined as multi-channel input for CNN-Transformer model.
"""

import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from scipy.ndimage import zoom
from typing import Tuple, Optional, Dict
import warnings

SAMPLE_RATE = 4000
TARGET_DURATION = 10.0
N_MELS = 128
N_FFT = 256
HOP_LENGTH = 64
N_MFCC = 13


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.std() == 0:
        return audio
    return (audio - audio.mean()) / audio.std()


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


def create_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
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


def create_mfcc_with_deltas(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    combined = np.vstack([mfcc, delta, delta2])
    return combined


def create_cwt_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_scales: int = N_MELS,
    wavelet: str = 'morl'
) -> np.ndarray:
    try:
        import pywt
        
        scales = np.arange(1, n_scales + 1)
        
        coefficients, frequencies = pywt.cwt(audio, scales, wavelet, 1.0/sr)
        
        cwt_spec = np.abs(coefficients)
        cwt_spec_db = librosa.power_to_db(cwt_spec ** 2, ref=np.max)
        
        return cwt_spec_db
        
    except ImportError:
        warnings.warn("pywt not installed. Using Mel spectrogram as fallback for CWT channel.")
        return create_mel_spectrogram(audio, sr, n_scales)


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    spec_min = spec.min()
    spec_max = spec.max()
    
    if spec_max - spec_min == 0:
        return np.zeros_like(spec)
    
    return (spec - spec_min) / (spec_max - spec_min)


def resize_spectrogram(spec: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    current_height, current_width = spec.shape
    zoom_h = target_height / current_height
    zoom_w = target_width / current_width
    
    resized = zoom(spec, (zoom_h, zoom_w), order=1)
    return resized


def create_single_channel_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_height: int = 128,
    target_width: int = 626
) -> np.ndarray:
    mel_spec = create_mel_spectrogram(audio, sr)
    mel_spec = normalize_spectrogram(mel_spec)
    mel_spec = resize_spectrogram(mel_spec, target_height, target_width)
    
    return mel_spec.astype(np.float32)[np.newaxis, :, :]


def create_three_channel_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_height: int = 128,
    target_width: int = 626
) -> np.ndarray:
    mel_spec = create_mel_spectrogram(audio, sr)
    delta = librosa.feature.delta(mel_spec, order=1)
    delta2 = librosa.feature.delta(mel_spec, order=2)
    
    mel_spec = normalize_spectrogram(mel_spec)
    delta = normalize_spectrogram(delta)
    delta2 = normalize_spectrogram(delta2)
    
    mel_spec = resize_spectrogram(mel_spec, target_height, target_width)
    delta = resize_spectrogram(delta, target_height, target_width)
    delta2 = resize_spectrogram(delta2, target_height, target_width)
    
    multichannel = np.stack([mel_spec, delta, delta2], axis=0)
    return multichannel.astype(np.float32)


def create_multi_channel_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_height: int = 128,
    target_width: int = 626,
    use_cwt: bool = True
) -> np.ndarray:
    mel_spec = create_mel_spectrogram(audio, sr)
    mel_spec = normalize_spectrogram(mel_spec)
    mel_spec = resize_spectrogram(mel_spec, target_height, target_width)
    
    mfcc_features = create_mfcc_with_deltas(audio, sr)
    mfcc_features = normalize_spectrogram(mfcc_features)
    mfcc_features = resize_spectrogram(mfcc_features, target_height, target_width)
    
    if use_cwt:
        cwt_spec = create_cwt_spectrogram(audio, sr, n_scales=target_height)
        cwt_spec = normalize_spectrogram(cwt_spec)
        cwt_spec = resize_spectrogram(cwt_spec, target_height, target_width)
    else:
        delta = librosa.feature.delta(create_mel_spectrogram(audio, sr), order=1)
        cwt_spec = normalize_spectrogram(delta)
        cwt_spec = resize_spectrogram(cwt_spec, target_height, target_width)
    
    multichannel = np.stack([mel_spec, mfcc_features, cwt_spec], axis=0)
    return multichannel.astype(np.float32)


def preprocess_audio_for_detection(
    file_path: str,
    target_duration: float = TARGET_DURATION,
    sr: int = SAMPLE_RATE,
    apply_filter: bool = True,
    multi_channel: bool = True,
    use_cwt: bool = False
) -> np.ndarray:
    audio = load_audio(file_path, sr)
    
    if apply_filter:
        audio = apply_bandpass_filter(audio, sr)
    
    audio = normalize_audio(audio)
    
    target_length = int(target_duration * sr)
    audio = pad_or_truncate(audio, target_length)
    
    if multi_channel:
        return create_multi_channel_features(audio, sr, use_cwt=use_cwt)
    else:
        return create_single_channel_features(audio, sr)


def preprocess_audio_array(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_duration: float = TARGET_DURATION,
    apply_filter: bool = True,
    multi_channel: bool = True,
    use_cwt: bool = False
) -> np.ndarray:
    if apply_filter:
        audio = apply_bandpass_filter(audio, sr)
    
    audio = normalize_audio(audio)
    
    target_length = int(target_duration * sr)
    audio = pad_or_truncate(audio, target_length)
    
    if multi_channel:
        return create_multi_channel_features(audio, sr, use_cwt=use_cwt)
    else:
        return create_single_channel_features(audio, sr)


if __name__ == "__main__":
    print("Multi-channel feature extraction module")
    print(f"Settings: SR={SAMPLE_RATE}, Duration={TARGET_DURATION}s, Mels={N_MELS}, MFCC={N_MFCC}")
