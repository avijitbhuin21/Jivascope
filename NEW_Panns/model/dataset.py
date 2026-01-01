"""
Dataset Classes for PANNs Heart Sound Classification.

Handles loading audio files and extracting mel spectrogram features for PANNs.
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.audio import load_audio, apply_bandpass_filter, normalize_audio, pad_or_truncate, AudioAugmentation


PANNS_SAMPLE_RATE = 32000
PANNS_TARGET_DURATION = 10.0
PANNS_N_MELS = 64
PANNS_N_FFT = 1024
PANNS_HOP_LENGTH = 320
PANNS_FMIN = 50
PANNS_FMAX = 14000


def extract_panns_features(
    audio: np.ndarray,
    sr: int = PANNS_SAMPLE_RATE,
    n_mels: int = PANNS_N_MELS,
    n_fft: int = PANNS_N_FFT,
    hop_length: int = PANNS_HOP_LENGTH,
    fmin: int = PANNS_FMIN,
    fmax: int = PANNS_FMAX
) -> torch.Tensor:
    """
    Extract log-mel spectrogram features for PANNs model.
    
    Args:
        audio: Audio waveform array
        sr: Sample rate (32kHz for PANNs)
        n_mels: Number of mel bins (64 for PANNs)
        n_fft: FFT window size
        hop_length: Hop length for STFT
        fmin: Minimum frequency
        fmax: Maximum frequency
    
    Returns:
        Log-mel spectrogram tensor of shape (time_steps, n_mels)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    log_mel = log_mel.T
    
    return torch.tensor(log_mel, dtype=torch.float32)


class PANNsHeartSoundDataset(Dataset):
    """Dataset for PANNs heart sound classification."""
    
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        augment: bool = False,
        augment_prob: float = 0.5,
        target_duration: float = PANNS_TARGET_DURATION,
        sample_rate: int = PANNS_SAMPLE_RATE
    ):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.augment = augment
        self.target_duration = target_duration
        self.sample_rate = sample_rate
        
        if augment:
            self.augmentation = AudioAugmentation(augment_prob=augment_prob)
        else:
            self.augmentation = None
        
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self) -> list:
        samples = []
        
        for idx, row in self.df.iterrows():
            patient_id = str(row['Patient ID'])
            outcome_label = int(row['Outcome_Label'])
            
            wav_file = f"{patient_id}.wav"
            file_path = os.path.join(self.data_dir, wav_file)
            
            if os.path.exists(file_path):
                samples.append({
                    'patient_id': patient_id,
                    'file_path': file_path,
                    'outcome_label': outcome_label,
                    'heart_sound_label': 1
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        audio = load_audio(sample['file_path'], target_sr=4000)
        
        if self.augmentation is not None:
            audio = self.augmentation(audio, 4000)
        
        audio = apply_bandpass_filter(audio, sr=4000)
        audio = normalize_audio(audio)
        
        target_length_4k = int(self.target_duration * 4000)
        audio = pad_or_truncate(audio, target_length_4k)
        
        audio_32k = librosa.resample(audio, orig_sr=4000, target_sr=self.sample_rate)
        
        features = extract_panns_features(audio_32k, sr=self.sample_rate)
        
        labels = torch.tensor(
            [sample['heart_sound_label'], sample['outcome_label']],
            dtype=torch.float32
        )
        
        return {
            'features': features,
            'labels': labels,
            'patient_id': sample['patient_id']
        }


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    train_dir: str,
    val_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    augment_train: bool = True,
    use_balanced_sampling: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = PANNsHeartSoundDataset(
        train_csv, train_dir, augment=augment_train
    )
    val_dataset = PANNsHeartSoundDataset(
        val_csv, val_dir, augment=False
    )
    
    sampler = None
    shuffle = True
    
    if use_balanced_sampling:
        labels = [sample['outcome_label'] for sample in train_dataset.samples]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
        print(f"Using balanced sampling: {class_counts[0]} Normal, {class_counts[1]} Abnormal")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    expected_time_steps = int(PANNS_TARGET_DURATION * PANNS_SAMPLE_RATE / PANNS_HOP_LENGTH) + 1
    print(f"PANNs Dataset settings:")
    print(f"  Sample Rate: {PANNS_SAMPLE_RATE} Hz")
    print(f"  Duration: {PANNS_TARGET_DURATION}s")
    print(f"  Mel Bins: {PANNS_N_MELS}")
    print(f"  Expected feature shape: ({expected_time_steps}, {PANNS_N_MELS})")
