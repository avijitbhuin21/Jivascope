"""
Configuration Management for Heart Sound Classification
Centralized configuration for all training and inference settings
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path
import os


@dataclass
class AudioConfig:
    sample_rate: int = 4000
    target_duration: float = 10.0
    apply_filter: bool = True
    filter_low_freq: float = 25.0
    filter_high_freq: float = 400.0


@dataclass
class SpectrogramConfig:
    n_mels: int = 128
    n_fft: int = 256
    hop_length: int = 64
    target_height: int = 128
    target_width: int = 313


@dataclass
class AugmentationConfig:
    time_shift_prob: float = 0.5
    time_shift_max: float = 0.1
    noise_prob: float = 0.4
    noise_snr_range: Tuple[int, int] = (25, 40)
    time_stretch_prob: float = 0.2
    time_stretch_range: Tuple[float, float] = (0.95, 1.05)
    volume_prob: float = 0.3
    volume_range: Tuple[float, float] = (0.85, 1.15)
    spec_freq_mask_prob: float = 0.5
    spec_time_mask_prob: float = 0.5


@dataclass
class ModelConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    num_murmur_classes: int = 2
    num_outcome_classes: int = 2
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    early_stopping_patience: int = 10
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    murmur_loss_weight: float = 0.6
    outcome_loss_weight: float = 0.4


@dataclass
class PathConfig:
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    dataset_dir: Path = field(default=None)
    cleaned_data_dir: Path = field(default=None)
    audio_dir: Path = field(default=None)
    checkpoint_dir: Path = field(default=None)
    log_dir: Path = field(default=None)
    
    def __post_init__(self):
        if self.dataset_dir is None:
            self.dataset_dir = self.base_dir / 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
        if self.cleaned_data_dir is None:
            self.cleaned_data_dir = self.base_dir / 'cleaned_data'
        if self.audio_dir is None:
            self.audio_dir = self.dataset_dir / 'training_data'
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.base_dir / 'checkpoints'
        if self.log_dir is None:
            self.log_dir = self.base_dir / 'logs'


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    recording_locations: List[str] = field(default_factory=lambda: ['AV', 'MV', 'PV', 'TV', 'Phc'])
    seed: int = 42
    num_workers: int = 0
    device: str = "cuda"


def get_default_config() -> Config:
    """
    Get default configuration.
    
    Returns:
        Default Config instance
    """
    return Config()


def print_config(config: Config) -> None:
    """
    Print configuration in readable format.
    
    Args:
        config: Config instance
    """
    print("=" * 60)
    print("JIVASCOPE CONFIGURATION")
    print("=" * 60)
    
    print(f"\n📁 Paths:")
    print(f"  Base: {config.paths.base_dir}")
    print(f"  Audio: {config.paths.audio_dir}")
    print(f"  Checkpoints: {config.paths.checkpoint_dir}")
    
    print(f"\n🎵 Audio:")
    print(f"  Sample Rate: {config.audio.sample_rate} Hz")
    print(f"  Duration: {config.audio.target_duration}s")
    print(f"  Bandpass Filter: {config.audio.apply_filter}")
    
    print(f"\n📊 Spectrogram:")
    print(f"  Mel Bands: {config.spectrogram.n_mels}")
    print(f"  FFT Size: {config.spectrogram.n_fft}")
    print(f"  Output Shape: ({config.spectrogram.target_height}, {config.spectrogram.target_width})")
    
    print(f"\n🤖 Model:")
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Pretrained: {config.model.pretrained}")
    print(f"  Classes: Murmur={config.model.num_murmur_classes}, Outcome={config.model.num_outcome_classes}")
    
    print(f"\n🏋️ Training:")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Loss Weights: Murmur={config.training.murmur_loss_weight}, Outcome={config.training.outcome_loss_weight}")
    
    print("=" * 60)


if __name__ == "__main__":
    config = get_default_config()
    print_config(config)
