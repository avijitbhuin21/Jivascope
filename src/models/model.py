"""
Heart Sound Classifier Model

Multi-task CNN model for heart sound classification with dual heads:
- Murmur Detection: Absent/Present (2-class)
- Clinical Outcome: Normal/Abnormal (2-class)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .backbone import create_backbone
from .classifier import ClassificationHead


class HeartSoundClassifier(nn.Module):
    """
    Multi-task model for heart sound classification.
    
    Architecture:
        Input (3, 128, W) → Backbone → Features → Dual Heads → Outputs
        
    Outputs:
        - murmur: (batch_size, num_murmur_classes) logits for Absent/Present
        - outcome: (batch_size, num_outcome_classes) logits for Normal/Abnormal
    
    Args:
        backbone_name: Pretrained backbone ('efficientnet_b0', 'resnet18', 'resnet34')
        pretrained: Whether to load ImageNet pretrained weights
        hidden_dim: Hidden dimension for classification heads
        num_murmur_classes: Number of murmur classes (default: 2)
        num_outcome_classes: Number of outcome classes (default: 2)
        dropout: Dropout probability for regularization
    """
    
    def __init__(
        self,
        backbone_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        hidden_dim: int = 512,
        num_murmur_classes: int = 2,
        num_outcome_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.hidden_dim = hidden_dim
        self.num_murmur_classes = num_murmur_classes
        self.num_outcome_classes = num_outcome_classes
        
        self.backbone, feature_dim = create_backbone(backbone_name, pretrained)
        self.feature_dim = feature_dim
        
        self.murmur_head = ClassificationHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_murmur_classes,
            dropout=dropout
        )
        
        self.outcome_head = ClassificationHead(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_outcome_classes,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input spectrogram of shape (batch_size, 3, height, width)
               Expected shape: (B, 3, 128, 313) for heart sound spectrograms
            
        Returns:
            Dictionary with:
                - 'murmur': Logits of shape (batch_size, num_murmur_classes)
                - 'outcome': Logits of shape (batch_size, num_outcome_classes)
        """
        features = self.backbone(x)
        
        murmur_logits = self.murmur_head(features)
        outcome_logits = self.outcome_head(features)
        
        return {
            'murmur': murmur_logits,
            'outcome': outcome_logits
        }
    
    def get_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get class predictions (not logits).
        
        Args:
            x: Input spectrogram
            
        Returns:
            Dictionary with predicted class indices
        """
        outputs = self.forward(x)
        return {
            'murmur': torch.argmax(outputs['murmur'], dim=1),
            'outcome': torch.argmax(outputs['outcome'], dim=1)
        }
    
    def get_probabilities(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get class probabilities (softmax applied).
        
        Args:
            x: Input spectrogram
            
        Returns:
            Dictionary with class probabilities
        """
        outputs = self.forward(x)
        return {
            'murmur': torch.softmax(outputs['murmur'], dim=1),
            'outcome': torch.softmax(outputs['outcome'], dim=1)
        }
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning only the heads."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters for full training."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_model(
    backbone_name: str = 'efficientnet_b0',
    pretrained: bool = True,
    hidden_dim: int = 512,
    num_murmur_classes: int = 2,
    num_outcome_classes: int = 2,
    dropout: float = 0.3,
    device: Optional[str] = None
) -> HeartSoundClassifier:
    """
    Factory function to create HeartSoundClassifier.
    
    Args:
        backbone_name: Pretrained backbone name
        pretrained: Whether to load pretrained weights
        hidden_dim: Hidden dimension for classification heads
        num_murmur_classes: Number of murmur classes
        num_outcome_classes: Number of outcome classes
        dropout: Dropout probability
        device: Device to place model on ('cuda', 'cpu', or None for auto)
        
    Returns:
        HeartSoundClassifier model
    """
    model = HeartSoundClassifier(
        backbone_name=backbone_name,
        pretrained=pretrained,
        hidden_dim=hidden_dim,
        num_murmur_classes=num_murmur_classes,
        num_outcome_classes=num_outcome_classes,
        dropout=dropout
    )
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    return model


def create_model_from_config(config) -> HeartSoundClassifier:
    """
    Create model from Config object.
    
    Args:
        config: Config object with model settings
        
    Returns:
        HeartSoundClassifier model
    """
    return create_model(
        backbone_name=config.model.backbone,
        pretrained=config.model.pretrained,
        hidden_dim=512,
        num_murmur_classes=config.model.num_murmur_classes,
        num_outcome_classes=config.model.num_outcome_classes,
        dropout=config.model.dropout,
        device=config.device
    )


def print_model_summary(model: HeartSoundClassifier) -> None:
    """
    Print model summary with architecture and parameter counts.
    
    Args:
        model: HeartSoundClassifier model
    """
    print("=" * 60)
    print("HEART SOUND CLASSIFIER - MODEL SUMMARY")
    print("=" * 60)
    
    print(f"\n📐 Architecture:")
    print(f"  Backbone: {model.backbone_name}")
    print(f"  Feature Dimension: {model.feature_dim}")
    print(f"  Hidden Dimension: {model.hidden_dim}")
    
    print(f"\n🎯 Classification Heads:")
    print(f"  Murmur: {model.num_murmur_classes} classes (Absent/Present)")
    print(f"  Outcome: {model.num_outcome_classes} classes (Normal/Abnormal)")
    
    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)
    
    print(f"\n📊 Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {total_params - trainable_params:,}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("Testing HeartSoundClassifier...")
    
    model = create_model(
        backbone_name='efficientnet_b0',
        pretrained=False,
        device='cpu'
    )
    
    print_model_summary(model)
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 313)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        predictions = model.get_predictions(x)
        probabilities = model.get_probabilities(x)
    
    print(f"\n🧪 Forward Pass Test:")
    print(f"  Input: {tuple(x.shape)}")
    print(f"  Murmur logits: {tuple(outputs['murmur'].shape)}")
    print(f"  Outcome logits: {tuple(outputs['outcome'].shape)}")
    print(f"  Murmur predictions: {predictions['murmur'].tolist()}")
    print(f"  Outcome predictions: {predictions['outcome'].tolist()}")
    
    print("\n✓ All tests passed!")
