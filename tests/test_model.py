"""
Test Suite for Model Architecture

Tests backbone creation, classification heads, and full model forward pass.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    HeartSoundClassifier,
    ClassificationHead,
    create_backbone,
    create_model,
    print_model_summary
)


class TestBackbone:
    """Tests for backbone module."""
    
    def test_efficientnet_b0_creation(self):
        """Test EfficientNet-B0 backbone creation."""
        model, feature_dim = create_backbone('efficientnet_b0', pretrained=False)
        assert feature_dim == 1280
        assert model is not None
        
    def test_resnet18_creation(self):
        """Test ResNet18 backbone creation."""
        model, feature_dim = create_backbone('resnet18', pretrained=False)
        assert feature_dim == 512
        assert model is not None
        
    def test_resnet34_creation(self):
        """Test ResNet34 backbone creation."""
        model, feature_dim = create_backbone('resnet34', pretrained=False)
        assert feature_dim == 512
        assert model is not None
        
    def test_efficientnet_b0_forward(self):
        """Test EfficientNet-B0 forward pass."""
        model, feature_dim = create_backbone('efficientnet_b0', pretrained=False)
        x = torch.randn(2, 3, 128, 313)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, feature_dim)
        
    def test_resnet18_forward(self):
        """Test ResNet18 forward pass."""
        model, feature_dim = create_backbone('resnet18', pretrained=False)
        x = torch.randn(2, 3, 128, 313)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, feature_dim)
        
    def test_invalid_backbone_raises_error(self):
        """Test that invalid backbone name raises ValueError."""
        with pytest.raises(ValueError):
            create_backbone('invalid_backbone')
            

class TestClassificationHead:
    """Tests for classification head module."""
    
    def test_head_creation(self):
        """Test classification head creation."""
        head = ClassificationHead(input_dim=1280, hidden_dim=512, num_classes=2)
        assert head is not None
        
    def test_head_forward_pass(self):
        """Test classification head forward pass."""
        head = ClassificationHead(input_dim=1280, hidden_dim=512, num_classes=2)
        x = torch.randn(4, 1280)
        out = head(x)
        assert out.shape == (4, 2)
        
    def test_head_different_classes(self):
        """Test heads with different number of classes."""
        head_2class = ClassificationHead(input_dim=512, hidden_dim=256, num_classes=2)
        head_3class = ClassificationHead(input_dim=512, hidden_dim=256, num_classes=3)
        
        x = torch.randn(2, 512)
        assert head_2class(x).shape == (2, 2)
        assert head_3class(x).shape == (2, 3)
        
    def test_head_dropout(self):
        """Test that dropout is applied in training mode."""
        head = ClassificationHead(input_dim=1280, hidden_dim=512, num_classes=2, dropout=0.5)
        head.train()
        
        x = torch.randn(10, 1280)
        outputs = [head(x) for _ in range(5)]
        
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should produce different outputs in training mode"


class TestHeartSoundClassifier:
    """Tests for the full HeartSoundClassifier model."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = create_model(
            backbone_name='efficientnet_b0',
            pretrained=False,
            device='cpu'
        )
        assert model is not None
        assert isinstance(model, HeartSoundClassifier)
        
    def test_model_forward_pass(self):
        """Test model forward pass with expected input dimensions."""
        model = create_model(pretrained=False, device='cpu')
        x = torch.randn(2, 3, 128, 313)
        
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            
        assert 'murmur' in outputs
        assert 'outcome' in outputs
        assert outputs['murmur'].shape == (2, 2)
        assert outputs['outcome'].shape == (2, 2)
        
    def test_model_predictions(self):
        """Test get_predictions method."""
        model = create_model(pretrained=False, device='cpu')
        x = torch.randn(4, 3, 128, 313)
        
        model.eval()
        with torch.no_grad():
            preds = model.get_predictions(x)
            
        assert preds['murmur'].shape == (4,)
        assert preds['outcome'].shape == (4,)
        assert all(p in [0, 1] for p in preds['murmur'].tolist())
        assert all(p in [0, 1] for p in preds['outcome'].tolist())
        
    def test_model_probabilities(self):
        """Test get_probabilities method."""
        model = create_model(pretrained=False, device='cpu')
        x = torch.randn(4, 3, 128, 313)
        
        model.eval()
        with torch.no_grad():
            probs = model.get_probabilities(x)
            
        assert probs['murmur'].shape == (4, 2)
        assert probs['outcome'].shape == (4, 2)
        
        murmur_sums = probs['murmur'].sum(dim=1)
        outcome_sums = probs['outcome'].sum(dim=1)
        assert torch.allclose(murmur_sums, torch.ones(4), atol=1e-5)
        assert torch.allclose(outcome_sums, torch.ones(4), atol=1e-5)
        
    def test_model_freeze_backbone(self):
        """Test freezing backbone parameters."""
        model = create_model(pretrained=False, device='cpu')
        
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
            
        for param in model.murmur_head.parameters():
            assert param.requires_grad
        for param in model.outcome_head.parameters():
            assert param.requires_grad
            
    def test_model_unfreeze_backbone(self):
        """Test unfreezing backbone parameters."""
        model = create_model(pretrained=False, device='cpu')
        model.freeze_backbone()
        model.unfreeze_backbone()
        
        for param in model.backbone.parameters():
            assert param.requires_grad
            
    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = create_model(
            backbone_name='efficientnet_b0',
            pretrained=False,
            device='cpu'
        )
        
        total_params = model.count_parameters(trainable_only=False)
        trainable_params = model.count_parameters(trainable_only=True)
        
        assert total_params > 0
        assert trainable_params == total_params
        
        assert total_params > 4_000_000
        assert total_params < 10_000_000
        
    def test_different_backbones(self):
        """Test model creation with different backbones."""
        for backbone in ['efficientnet_b0', 'resnet18', 'resnet34']:
            model = create_model(
                backbone_name=backbone,
                pretrained=False,
                device='cpu'
            )
            
            x = torch.randn(1, 3, 128, 313)
            model.eval()
            with torch.no_grad():
                outputs = model(x)
                
            assert outputs['murmur'].shape == (1, 2)
            assert outputs['outcome'].shape == (1, 2)
            
    def test_model_eval_mode(self):
        """Test model behaves consistently in eval mode."""
        model = create_model(pretrained=False, device='cpu')
        model.eval()
        
        x = torch.randn(2, 3, 128, 313)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
            
        assert torch.allclose(out1['murmur'], out2['murmur'])
        assert torch.allclose(out1['outcome'], out2['outcome'])


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test complete forward pass with realistic dimensions."""
        model = create_model(
            backbone_name='efficientnet_b0',
            pretrained=False,
            hidden_dim=512,
            num_murmur_classes=2,
            num_outcome_classes=2,
            dropout=0.3,
            device='cpu'
        )
        
        batch_size = 8
        x = torch.randn(batch_size, 3, 128, 313)
        
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            predictions = model.get_predictions(x)
            probabilities = model.get_probabilities(x)
            
        assert outputs['murmur'].shape == (batch_size, 2)
        assert outputs['outcome'].shape == (batch_size, 2)
        assert predictions['murmur'].shape == (batch_size,)
        assert predictions['outcome'].shape == (batch_size,)
        assert probabilities['murmur'].shape == (batch_size, 2)
        assert probabilities['outcome'].shape == (batch_size, 2)
        
    def test_model_summary(self, capsys):
        """Test model summary printing."""
        model = create_model(pretrained=False, device='cpu')
        print_model_summary(model)
        
        captured = capsys.readouterr()
        assert 'HEART SOUND CLASSIFIER' in captured.out
        assert 'efficientnet_b0' in captured.out
        assert 'Parameters' in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
