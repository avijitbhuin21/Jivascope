"""
Models Module for Heart Sound Classification

Exports:
    - HeartSoundClassifier: Main multi-task classification model
    - ClassificationHead: Classification head module
    - create_backbone: Backbone factory function
    - create_model: Model factory function
    - create_model_from_config: Create model from Config object
    - print_model_summary: Print model architecture summary
"""

from .backbone import create_backbone, get_backbone_info
from .classifier import ClassificationHead
from .model import (
    HeartSoundClassifier,
    create_model,
    create_model_from_config,
    print_model_summary
)

__all__ = [
    'HeartSoundClassifier',
    'ClassificationHead',
    'create_backbone',
    'get_backbone_info',
    'create_model',
    'create_model_from_config',
    'print_model_summary'
]
