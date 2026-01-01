"""
AST (Audio Spectrogram Transformer) Model for Heart Sound Classification.

Uses pre-trained AST model from Hugging Face and fine-tunes for heart sound classification.
"""

import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig


class ASTHeartClassifier(nn.Module):
    """
    AST-based classifier for heart sound detection.
    
    Outputs two logits:
    - heart_sound_present (index 0)
    - murmur_present (index 1)
    """
    
    def __init__(
        self,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_classes: int = 2,
        hidden_size: int = 768,
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.ast = ASTModel.from_pretrained(
            pretrained_model,
            low_cpu_mem_usage=True
        )
        
        if freeze_encoder:
            for param in self.ast.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_values: Mel spectrogram features of shape (batch, time_steps, mel_bins)
                         Expected: (batch, 1024, 128) for 10s audio at 16kHz
        
        Returns:
            Logits of shape (batch, 2) for [heart_sound, murmur]
        """
        outputs = self.ast(input_values=input_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits
    
    def get_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        """Get embeddings before classification head."""
        outputs = self.ast(input_values=input_values)
        return outputs.pooler_output


class ASTHeartClassifierFromScratch(nn.Module):
    """
    AST-based classifier trained from scratch (no pre-trained weights).
    Useful for comparison or when pre-trained weights aren't suitable.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        max_length: int = 1024,
        frequency_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        config = ASTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_length=max_length,
            frequency_size=frequency_size,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        
        self.ast = ASTModel(config)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.ast(input_values=input_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def create_model(
    model_type: str = 'pretrained',
    pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_classes: int = 2,
    freeze_encoder: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create AST model.
    
    Args:
        model_type: 'pretrained' or 'scratch'
        pretrained_model: HuggingFace model name for pre-trained AST
        num_classes: Number of output classes (default 2: heart_sound, murmur)
        freeze_encoder: Whether to freeze the AST encoder weights
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        AST model ready for training
    """
    if model_type == 'pretrained':
        model = ASTHeartClassifier(
            pretrained_model=pretrained_model,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            **kwargs
        )
    elif model_type == 'scratch':
        model = ASTHeartClassifierFromScratch(
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"AST Model created: {total_params:,} params ({trainable_params:,} trainable)")
    
    return model
