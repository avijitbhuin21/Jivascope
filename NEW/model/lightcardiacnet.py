"""
LightCardiacNet Model Architecture

Lightweight Bi-GRU ensemble with attention for heart sound classification.
Outputs: [heart_sound_present, murmur_present]
"""

import torch
import torch.nn as nn
from .attention import AttentionLayer


class BiGRUNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = 39,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = AttentionLayer(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
        rnn_out, _ = self.bigru(x)
        context, attn_weights = self.attention(rnn_out)
        logits = self.classifier(context)
        return logits, attn_weights


class LightCardiacNet(nn.Module):
    def __init__(
        self,
        input_size: int = 39,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.network1 = BiGRUNetwork(input_size, hidden_size, num_layers, num_classes, dropout)
        self.network2 = BiGRUNetwork(input_size, hidden_size, num_layers, num_classes, dropout)
        
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> tuple:
        logits1, attn1 = self.network1(x)
        logits2, attn2 = self.network2(x)
        
        w = torch.sigmoid(self.ensemble_weight)
        ensemble_logits = w * logits1 + (1 - w) * logits2
        
        return ensemble_logits, (attn1, attn2)
    
    def get_predictions(self, logits: torch.Tensor, threshold: float = 0.5) -> dict:
        probs = torch.sigmoid(logits)
        
        heart_sound_prob = probs[:, 0]
        murmur_prob = probs[:, 1]
        
        heart_sound_present = heart_sound_prob > threshold
        murmur_present = (murmur_prob > threshold) & heart_sound_present
        
        return {
            'heart_sound_present': heart_sound_present,
            'murmur_present': murmur_present,
            'heart_sound_prob': heart_sound_prob,
            'murmur_prob': murmur_prob
        }


class LightCardiacNetSingle(nn.Module):
    def __init__(
        self,
        input_size: int = 39,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.network = BiGRUNetwork(input_size, hidden_size, num_layers, num_classes, dropout)
    
    def forward(self, x: torch.Tensor) -> tuple:
        return self.network(x)


def create_model(
    model_type: str = 'ensemble',
    input_size: int = 39,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_classes: int = 2,
    dropout: float = 0.3
) -> nn.Module:
    if model_type == 'ensemble':
        return LightCardiacNet(input_size, hidden_size, num_layers, num_classes, dropout)
    elif model_type == 'single':
        return LightCardiacNetSingle(input_size, hidden_size, num_layers, num_classes, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    model = LightCardiacNet()
    x = torch.randn(4, 625, 39)
    logits, (attn1, attn2) = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attn1.shape}, {attn2.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
