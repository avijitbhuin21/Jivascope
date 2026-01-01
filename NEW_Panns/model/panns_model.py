"""
PANNs (Pre-trained Audio Neural Networks) Model for Heart Sound Classification.

Uses Cnn14 architecture with pre-trained AudioSet weights.
Reference: https://github.com/qiuqiangkong/panns_inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weight()
    
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise ValueError(f"Unknown pool type: {pool_type}")
        
        return x


class Cnn14(nn.Module):
    """
    Cnn14 architecture from PANNs.
    
    Reference: Kong, Qiuqiang, et al. "PANNs: Large-Scale Pretrained Audio Neural Networks 
    for Audio Pattern Recognition." IEEE/ACM Transactions on Audio, Speech, and Language 
    Processing 28 (2020): 2880-2894.
    """
    
    def __init__(
        self,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        num_classes: int = 527
    ):
        super().__init__()
        
        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, num_classes, bias=True)
        
        self.init_weight()
    
    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram of shape (batch, time_steps, mel_bins)
        
        Returns:
            Embeddings of shape (batch, 2048)
        """
        x = x.transpose(1, 2).unsqueeze(-1)
        x = self.bn0(x)
        x = x.squeeze(-1).unsqueeze(1)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        
        return x


class PANNsHeartClassifier(nn.Module):
    """
    PANNs-based classifier for heart sound detection.
    
    Outputs two logits:
    - heart_sound_present (index 0)
    - murmur_present (index 1)
    """
    
    def __init__(
        self,
        pretrained_path: str = None,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cnn14 = Cnn14(num_classes=527)
        
        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)
        
        if freeze_encoder:
            for param in self.cnn14.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
    
    def _load_pretrained(self, pretrained_path: str):
        """Load pretrained Cnn14 weights."""
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        model_state = self.cnn14.state_dict()
        for key in list(state_dict.keys()):
            if key in model_state and state_dict[key].shape == model_state[key].shape:
                model_state[key] = state_dict[key]
        
        self.cnn14.load_state_dict(model_state)
        print("Loaded pretrained Cnn14 weights")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram features of shape (batch, time_steps, mel_bins)
        
        Returns:
            Logits of shape (batch, 2) for [heart_sound, murmur]
        """
        embeddings = self.cnn14(x)
        logits = self.classifier(embeddings)
        return logits
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings before classification head."""
        return self.cnn14(x)


class PANNsHeartClassifierLight(nn.Module):
    """
    Lighter version of PANNs classifier (fewer layers).
    Useful for faster training and inference on CPU.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.bn0 = nn.BatchNorm2d(64)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        self.num_classes = num_classes
        self._init_weights()
    
    def _init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).unsqueeze(-1)
        x = self.bn0(x)
        x = x.squeeze(-1).unsqueeze(1)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)


        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


def create_model(
    model_type: str = 'cnn14',
    pretrained_path: str = None,
    num_classes: int = 2,
    freeze_encoder: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create PANNs model.
    
    Args:
        model_type: 'cnn14' (full) or 'light' (smaller version)
        pretrained_path: Path to pretrained Cnn14 weights (optional)
        num_classes: Number of output classes (default 2: heart_sound, murmur)
        freeze_encoder: Whether to freeze the encoder weights
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        PANNs model ready for training
    """
    if model_type == 'cnn14':
        model = PANNsHeartClassifier(
            pretrained_path=pretrained_path,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            **kwargs
        )
    elif model_type == 'light':
        model = PANNsHeartClassifierLight(
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PANNs Model created: {total_params:,} params ({trainable_params:,} trainable)")
    
    return model
