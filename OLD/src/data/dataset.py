"""
PyTorch Dataset for Heart Sound Classification
Loads audio files and converts them to spectrograms for model input
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import os

from .preprocessing import (
    load_audio,
    normalize_audio,
    pad_or_truncate,
    apply_bandpass_filter,
    create_multichannel_spectrogram,
    SAMPLE_RATE,
    TARGET_DURATION
)
from .augmentation import AudioAugmentor, SpecAugment, get_train_augmentor, get_spec_augmentor


class HeartSoundDataset(Dataset):
    """
    PyTorch Dataset for heart sound classification.
    
    Each sample returns:
        - spectrogram: (3, H, W) tensor (RGB-like for pretrained CNN)
        - murmur_label: int (0=Absent, 1=Present)
        - outcome_label: int (0=Normal, 1=Abnormal)
        - patient_id: str (for analysis)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        is_training: bool = False,
        target_duration: float = TARGET_DURATION,
        sample_rate: int = SAMPLE_RATE,
        apply_filter: bool = True,
        audio_augmentor: Optional[AudioAugmentor] = None,
        spec_augmentor: Optional[SpecAugment] = None,
        recording_locations: Optional[List[str]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame with patient data (must have Patient ID, Murmur_Label, Outcome_Label)
            audio_dir: Directory containing audio (.wav) files
            is_training: Whether this is training data (enables augmentation)
            target_duration: Target audio duration in seconds
            sample_rate: Target sample rate
            apply_filter: Whether to apply bandpass filter
            audio_augmentor: Audio augmentation instance (training only)
            spec_augmentor: SpecAugment instance (training only)
            recording_locations: List of auscultation locations to use (e.g., ['AV', 'MV', 'PV', 'TV', 'Phc'])
        """
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.is_training = is_training
        self.target_duration = target_duration
        self.sample_rate = sample_rate
        self.apply_filter = apply_filter
        self.target_length = int(target_duration * sample_rate)
        
        if is_training:
            self.audio_augmentor = audio_augmentor or get_train_augmentor()
            self.spec_augmentor = spec_augmentor or get_spec_augmentor()
        else:
            self.audio_augmentor = None
            self.spec_augmentor = None
        
        self.recording_locations = recording_locations or ['AV', 'MV', 'PV', 'TV', 'Phc']
        
        self.samples = self._build_sample_list()
    
    def _build_sample_list(self) -> List[Dict]:
        """
        Build list of all audio samples with their labels.
        Each patient may have multiple recording locations.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        for idx, row in self.df.iterrows():
            patient_id = row['Patient ID']
            murmur_label = row['Murmur_Label']
            outcome_label = row['Outcome_Label']
            
            for loc in self.recording_locations:
                wav_path = self.audio_dir / f"{patient_id}_{loc}.wav"
                if wav_path.exists():
                    samples.append({
                        'patient_id': patient_id,
                        'location': loc,
                        'wav_path': str(wav_path),
                        'murmur_label': murmur_label,
                        'outcome_label': outcome_label
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (spectrogram, murmur_label, outcome_label, patient_id)
        """
        sample = self.samples[idx]
        
        audio = load_audio(sample['wav_path'], self.sample_rate)
        
        if self.apply_filter:
            audio = apply_bandpass_filter(audio, self.sample_rate)
        
        audio = normalize_audio(audio)
        
        audio = pad_or_truncate(audio, self.target_length)
        
        if self.is_training and self.audio_augmentor is not None:
            audio = self.audio_augmentor(audio, self.sample_rate)
        
        spectrogram = create_multichannel_spectrogram(
            audio, 
            self.sample_rate,
            target_height=128,
            target_width=313
        )
        
        if self.is_training and self.spec_augmentor is not None:
            spectrogram = self.spec_augmentor(spectrogram)
        
        spectrogram = torch.from_numpy(spectrogram).float()
        
        return (
            spectrogram,
            sample['murmur_label'],
            sample['outcome_label'],
            sample['patient_id']
        )


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    audio_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        audio_dir: Directory containing audio files
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments for HeartSoundDataset
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = HeartSoundDataset(
        train_df, audio_dir, is_training=True, **dataset_kwargs
    )
    val_dataset = HeartSoundDataset(
        val_df, audio_dir, is_training=False, **dataset_kwargs
    )
    test_dataset = HeartSoundDataset(
        test_df, audio_dir, is_training=False, **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights(df: pd.DataFrame) -> Dict[str, torch.Tensor]:
    """
    Calculate class weights for handling imbalanced data.
    
    Args:
        df: Training DataFrame
    
    Returns:
        Dictionary with murmur_weights and outcome_weights tensors
    """
    murmur_counts = df['Murmur_Label'].value_counts().sort_index()
    murmur_weights = len(df) / (2 * murmur_counts)
    
    outcome_counts = df['Outcome_Label'].value_counts().sort_index()
    outcome_weights = len(df) / (2 * outcome_counts)
    
    return {
        'murmur_weights': torch.tensor(murmur_weights.values, dtype=torch.float32),
        'outcome_weights': torch.tensor(outcome_weights.values, dtype=torch.float32)
    }


if __name__ == "__main__":
    print("Dataset module loaded successfully")
    print(f"Default settings: SR={SAMPLE_RATE}, Duration={TARGET_DURATION}s")
