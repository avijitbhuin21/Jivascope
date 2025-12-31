"""
Backbone Module for Heart Sound Classification

Provides pretrained CNN backbones for feature extraction.
Supported architectures: EfficientNet-B0, ResNet18, ResNet34
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


def create_backbone(name: str = 'efficientnet_b0', pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Create pretrained backbone for feature extraction.
    
    Args:
        name: Backbone architecture name. Options:
            - 'efficientnet_b0': 1280-dim output, ~5.3M params (recommended)
            - 'resnet18': 512-dim output, ~11.7M params
            - 'resnet34': 512-dim output, ~21.8M params
        pretrained: Whether to load ImageNet pretrained weights
        
    Returns:
        Tuple of (model, feature_dim) where:
            - model: nn.Module backbone with classifier removed
            - feature_dim: Output feature dimension
            
    Raises:
        ValueError: If backbone name is not supported
    """
    name = name.lower()
    
    if name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
        feature_dim = 1280
        
    elif name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        feature_dim = 512
        
    elif name == 'resnet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = nn.Identity()
        feature_dim = 512
        
    elif name == 'efficientnet_b3':
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier = nn.Identity()
        feature_dim = 1536
        
    else:
        supported = ['efficientnet_b0', 'efficientnet_b3', 'resnet18', 'resnet34']
        raise ValueError(f"Unsupported backbone: '{name}'. Supported: {supported}")
    
    return model, feature_dim


def get_backbone_info(name: str) -> dict:
    """
    Get information about a backbone architecture.
    
    Args:
        name: Backbone architecture name
        
    Returns:
        Dictionary with backbone information
    """
    info = {
        'efficientnet_b0': {
            'feature_dim': 1280,
            'params': '5.3M',
            'recommended': True,
            'description': 'Efficient architecture with good accuracy/speed tradeoff'
        },
        'efficientnet_b3': {
            'feature_dim': 1536,
            'params': '12M',
            'recommended': False,
            'description': 'Larger EfficientNet variant for higher accuracy'
        },
        'resnet18': {
            'feature_dim': 512,
            'params': '11.7M',
            'recommended': False,
            'description': 'Classic ResNet architecture, good baseline'
        },
        'resnet34': {
            'feature_dim': 512,
            'params': '21.8M',
            'recommended': False,
            'description': 'Deeper ResNet variant'
        }
    }
    
    name = name.lower()
    if name not in info:
        raise ValueError(f"Unknown backbone: '{name}'")
    
    return info[name]


if __name__ == "__main__":
    print("Testing backbone creation...")
    
    for backbone_name in ['efficientnet_b0', 'resnet18', 'resnet34']:
        model, feature_dim = create_backbone(backbone_name, pretrained=False)
        
        x = torch.randn(2, 3, 128, 313)
        with torch.no_grad():
            out = model(x)
        
        print(f"✓ {backbone_name}: input {tuple(x.shape)} → output {tuple(out.shape)} (feature_dim={feature_dim})")
    
    print("\nAll backbones tested successfully!")
