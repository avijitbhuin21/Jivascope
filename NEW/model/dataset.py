"""
Dataset Classes for Heart Sound Classification

Handles loading audio files and extracting features for training.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List
import numpy as np

from .features import (
    load_audio,
    apply_bandpass_filter,
    normalize_audio,
    pad_or_truncate,
    extract_features_for_bigru,
    extract_features_from_array,
    AudioAugmentation,
    SAMPLE_RATE,
    TARGET_DURATION
)


class HeartSoundDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        augment: bool = False,
        augment_prob: float = 0.5
    ):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.augment = augment
        
        if augment:
            self.augmentation = AudioAugmentation(augment_prob=augment_prob)
        else:
            self.augmentation = None
        
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self) -> List[dict]:
        samples = []
        
        for idx, row in self.df.iterrows():
            patient_id = str(row['Patient ID'])
            murmur_label = int(row['Murmur_Label'])
            
            wav_files = [f for f in os.listdir(self.data_dir) 
                        if f.startswith(f"{patient_id}_") and f.endswith('.wav')]
            
            for wav_file in wav_files:
                samples.append({
                    'patient_id': patient_id,
                    'file_path': os.path.join(self.data_dir, wav_file),
                    'valve': wav_file.replace(f"{patient_id}_", "").replace(".wav", ""),
                    'murmur_label': murmur_label,
                    'heart_sound_label': 1
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        audio = load_audio(sample['file_path'], SAMPLE_RATE)
        
        if self.augmentation is not None:
            audio = self.augmentation(audio, SAMPLE_RATE)
        
        audio = apply_bandpass_filter(audio, SAMPLE_RATE)
        audio = normalize_audio(audio)
        
        target_length = int(TARGET_DURATION * SAMPLE_RATE)
        audio = pad_or_truncate(audio, target_length)
        
        features = extract_features_from_array(audio, apply_filter=False)
        
        labels = torch.tensor(
            [sample['heart_sound_label'], sample['murmur_label']],
            dtype=torch.float32
        )
        
        return features, labels


class HeartSoundDatasetPatientLevel(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        max_valves: int = 4,
        augment: bool = False
    ):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.max_valves = max_valves
        self.augment = augment
        
        if augment:
            self.augmentation = AudioAugmentation()
        else:
            self.augmentation = None
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        patient_id = str(row['Patient ID'])
        murmur_label = int(row['Murmur_Label'])
        
        wav_files = [f for f in os.listdir(self.data_dir) 
                    if f.startswith(f"{patient_id}_") and f.endswith('.wav')]
        
        features_list = []
        for wav_file in wav_files[:self.max_valves]:
            file_path = os.path.join(self.data_dir, wav_file)
            
            audio = load_audio(file_path, SAMPLE_RATE)
            
            if self.augmentation is not None:
                audio = self.augmentation(audio, SAMPLE_RATE)
            
            features = extract_features_from_array(audio)
            features_list.append(features)
        
        while len(features_list) < self.max_valves:
            features_list.append(torch.zeros_like(features_list[0]))
        
        stacked_features = torch.stack(features_list)
        
        labels = torch.tensor([1, murmur_label], dtype=torch.float32)
        
        return stacked_features, labels, len(wav_files)


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = HeartSoundDataset(
        train_csv, train_dir, augment=augment_train
    )
    val_dataset = HeartSoundDataset(
        val_csv, val_dir, augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    print("Dataset module for heart sound classification")
