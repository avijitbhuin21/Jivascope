"""
LightCardiacNet Model Package

Heart Murmur Detection using Bi-GRU Ensemble with Attention
"""

from .lightcardiacnet import LightCardiacNet, BiGRUNetwork, create_model
from .attention import AttentionLayer
from .features import extract_features_for_bigru, AudioAugmentation
from .dataset import HeartSoundDataset, create_dataloaders
from .losses import WeightedBCELoss, FocalLoss, create_loss_function
from .trainer import Trainer, EarlyStopping
from .predictor import HeartSoundPredictor, load_predictor

__version__ = "1.0.0"
__all__ = [
    "LightCardiacNet",
    "BiGRUNetwork",
    "create_model",
    "AttentionLayer",
    "extract_features_for_bigru",
    "AudioAugmentation",
    "HeartSoundDataset",
    "create_dataloaders",
    "WeightedBCELoss",
    "FocalLoss",
    "create_loss_function",
    "Trainer",
    "EarlyStopping",
    "HeartSoundPredictor",
    "load_predictor"
]
