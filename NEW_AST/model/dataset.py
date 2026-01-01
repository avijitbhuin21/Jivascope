"""
Dataset Classes for AST Heart Sound Classification.

Handles loading audio files and extracting mel spectrogram features for AST.
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Tuple, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.audio import load_audio, apply_bandpass_filter, normalize_audio, pad_or_truncate, AudioAugmentation


AST_SAMPLE_RATE = 16000
AST_TARGET_DURATION = 10.0
AST_N_MELS = 128
AST_N_FFT = 400
AST_HOP_LENGTH = 160
AST_MAX_LENGTH = 1024


def extract_ast_features(
    audio: np.ndarray,
    sr: int = AST_SAMPLE_RATE,
    n_mels: int = AST_N_MELS,
    n_fft: int = AST_N_FFT,
    hop_length: int = AST_HOP_LENGTH,
    max_length: int = AST_MAX_LENGTH
) -> torch.Tensor:
    """
    Extract log-mel spectrogram features for AST model.
    
    Args:
        audio: Audio waveform array
        sr: Sample rate (16kHz for AST)
        n_mels: Number of mel bins (128 for AST)
        n_fft: FFT window size
        hop_length: Hop length for STFT
        max_length: Target time steps to match pretrained model (1214 for MIT AST)
    
    Returns:
        Log-mel spectrogram tensor of shape (max_length, n_mels)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=sr // 2
    )
    
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    log_mel = log_mel.T
    
    current_length = log_mel.shape[0]
    if current_length < max_length:
        padding = np.zeros((max_length - current_length, n_mels))
        log_mel = np.vstack([log_mel, padding])
    elif current_length > max_length:
        log_mel = log_mel[:max_length, :]
    
    return torch.tensor(log_mel, dtype=torch.float32)


class ASTHeartSoundDataset(Dataset):
    """Dataset for AST heart sound classification."""
    
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        augment: bool = False,
        augment_prob: float = 0.5,
        target_duration: float = AST_TARGET_DURATION,
        sample_rate: int = AST_SAMPLE_RATE
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
        
        audio_16k = librosa.resample(audio, orig_sr=4000, target_sr=self.sample_rate)
        
        features = extract_ast_features(audio_16k, sr=self.sample_rate)
        
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
    
    train_dataset = ASTHeartSoundDataset(
        train_csv, train_dir, augment=augment_train
    )
    val_dataset = ASTHeartSoundDataset(
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
    expected_time_steps = int(AST_TARGET_DURATION * AST_SAMPLE_RATE / AST_HOP_LENGTH) + 1
    print(f"AST Dataset settings:")
    print(f"  Sample Rate: {AST_SAMPLE_RATE} Hz")
    print(f"  Duration: {AST_TARGET_DURATION}s")
    print(f"  Mel Bins: {AST_N_MELS}")
    print(f"  Expected feature shape: ({expected_time_steps}, {AST_N_MELS})")
