"""
Classification Head Module for Heart Sound Classification

Provides classification head with FC layers, ReLU activation, and dropout regularization.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Single classification head with dropout regularization.
    
    Architecture:
        input → FC(hidden_dim) → ReLU → Dropout → FC(num_classes) → output
    
    Args:
        input_dim: Input feature dimension from backbone
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        dropout: Dropout probability for regularization
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 512, 
        num_classes: int = 2, 
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    print("Testing ClassificationHead...")
    
    batch_size = 4
    input_dim = 1280
    hidden_dim = 512
    num_classes = 2
    
    head = ClassificationHead(input_dim, hidden_dim, num_classes, dropout=0.3)
    
    x = torch.randn(batch_size, input_dim)
    out = head(x)
    
    print(f"✓ Input: {tuple(x.shape)} → Output: {tuple(out.shape)}")
    print(f"✓ Expected: ({batch_size}, {num_classes})")
    
    total_params = sum(p.numel() for p in head.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    print("\nClassificationHead test passed!")
