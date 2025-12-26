"""
Audio Data Augmentation for Heart Sound Classification
Implements various audio augmentation techniques for training
"""

import numpy as np
from typing import Optional, List, Callable
import random


class AudioAugmentor:
    """
    Audio augmentation pipeline for training data.
    Applies random augmentations to audio signals.
    """
    
    def __init__(
        self,
        time_shift_prob: float = 0.5,
        time_shift_max: float = 0.2,
        noise_prob: float = 0.5,
        noise_snr_range: tuple = (20, 40),
        time_stretch_prob: float = 0.3,
        time_stretch_range: tuple = (0.9, 1.1),
        pitch_shift_prob: float = 0.3,
        pitch_shift_range: tuple = (-2, 2),
        volume_prob: float = 0.3,
        volume_range: tuple = (0.8, 1.2)
    ):
        self.time_shift_prob = time_shift_prob
        self.time_shift_max = time_shift_max
        self.noise_prob = noise_prob
        self.noise_snr_range = noise_snr_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.volume_prob = volume_prob
        self.volume_range = volume_range
    
    def __call__(self, audio: np.ndarray, sr: int = 4000) -> np.ndarray:
        """
        Apply random augmentations to audio.
        
        Args:
            audio: Input audio signal
            sr: Sample rate
        
        Returns:
            Augmented audio signal
        """
        if random.random() < self.time_shift_prob:
            audio = self.time_shift(audio)
        
        if random.random() < self.noise_prob:
            audio = self.add_noise(audio)
        
        if random.random() < self.volume_prob:
            audio = self.change_volume(audio)
        
        if random.random() < self.time_stretch_prob:
            audio = self.time_stretch(audio, sr)
        
        return audio
    
    def time_shift(self, audio: np.ndarray) -> np.ndarray:
        """
        Randomly shift audio in time.
        
        Args:
            audio: Input audio
        
        Returns:
            Time-shifted audio
        """
        shift_max = int(len(audio) * self.time_shift_max)
        shift = random.randint(-shift_max, shift_max)
        
        if shift > 0:
            audio = np.pad(audio[shift:], (0, shift), mode='constant')
        elif shift < 0:
            audio = np.pad(audio[:shift], (-shift, 0), mode='constant')
        
        return audio
    
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise at random SNR level.
        
        Args:
            audio: Input audio
        
        Returns:
            Noisy audio
        """
        snr_db = random.uniform(*self.noise_snr_range)
        
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        
        return (audio + noise).astype(np.float32)
    
    def change_volume(self, audio: np.ndarray) -> np.ndarray:
        """
        Randomly change volume.
        
        Args:
            audio: Input audio
        
        Returns:
            Volume-adjusted audio
        """
        gain = random.uniform(*self.volume_range)
        return (audio * gain).astype(np.float32)
    
    def time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Time stretch audio without changing pitch.
        
        Args:
            audio: Input audio
            sr: Sample rate
        
        Returns:
            Time-stretched audio
        """
        try:
            import librosa
            rate = random.uniform(*self.time_stretch_range)
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            elif len(stretched) < len(audio):
                stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
            
            return stretched.astype(np.float32)
        except Exception:
            return audio


class SpecAugment:
    """
    SpecAugment: Augmentation on spectrograms.
    Applies time and frequency masking directly on spectrograms.
    """
    
    def __init__(
        self,
        freq_mask_prob: float = 0.5,
        freq_mask_param: int = 20,
        time_mask_prob: float = 0.5,
        time_mask_param: int = 50,
        num_freq_masks: int = 2,
        num_time_masks: int = 2
    ):
        self.freq_mask_prob = freq_mask_prob
        self.freq_mask_param = freq_mask_param
        self.time_mask_prob = time_mask_prob
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
    
    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram (C, H, W) or (H, W)
        
        Returns:
            Augmented spectrogram
        """
        spec = spectrogram.copy()
        
        if spec.ndim == 3:
            for c in range(spec.shape[0]):
                spec[c] = self._augment_single(spec[c])
        else:
            spec = self._augment_single(spec)
        
        return spec
    
    def _augment_single(self, spec: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to single-channel spectrogram.
        
        Args:
            spec: Single-channel spectrogram (H, W)
        
        Returns:
            Augmented spectrogram
        """
        freq_bins, time_frames = spec.shape
        
        if random.random() < self.freq_mask_prob:
            for _ in range(self.num_freq_masks):
                f = random.randint(0, min(self.freq_mask_param, freq_bins - 1))
                f0 = random.randint(0, freq_bins - f)
                spec[f0:f0 + f, :] = 0
        
        if random.random() < self.time_mask_prob:
            for _ in range(self.num_time_masks):
                t = random.randint(0, min(self.time_mask_param, time_frames - 1))
                t0 = random.randint(0, time_frames - t)
                spec[:, t0:t0 + t] = 0
        
        return spec


def get_train_augmentor() -> AudioAugmentor:
    """
    Get default augmentor for training.
    
    Returns:
        Configured AudioAugmentor instance
    """
    return AudioAugmentor(
        time_shift_prob=0.5,
        time_shift_max=0.1,
        noise_prob=0.4,
        noise_snr_range=(25, 40),
        time_stretch_prob=0.2,
        time_stretch_range=(0.95, 1.05),
        volume_prob=0.3,
        volume_range=(0.85, 1.15)
    )


def get_spec_augmentor() -> SpecAugment:
    """
    Get default SpecAugment for training.
    
    Returns:
        Configured SpecAugment instance
    """
    return SpecAugment(
        freq_mask_prob=0.5,
        freq_mask_param=15,
        time_mask_prob=0.5,
        time_mask_param=40,
        num_freq_masks=2,
        num_time_masks=2
    )


if __name__ == "__main__":
    print("Augmentation module loaded successfully")
    aug = get_train_augmentor()
    spec_aug = get_spec_augmentor()
    print(f"Audio augmentor: {aug}")
    print(f"SpecAugment: {spec_aug}")
