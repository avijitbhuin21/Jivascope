"""
Check AST model expected input shape.
"""

from transformers import ASTModel, ASTFeatureExtractor
import torch

print("Loading AST model config...")
model = ASTModel.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
print(f"Model config:")
print(f"  max_length: {model.config.max_length}")
print(f"  frequency_size: {model.config.frequency_size}")
print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
print(f"  hidden_size: {model.config.hidden_size}")

print("\nPosition embeddings shape:", model.embeddings.position_embeddings.shape)

print("\nLoading feature extractor...")
feature_extractor = ASTFeatureExtractor.from_pretrained('MIT/ast-finetuned-audioset-10-10-0.4593')
print(f"Feature extractor config:")
print(f"  sampling_rate: {feature_extractor.sampling_rate}")
print(f"  max_length: {feature_extractor.max_length}")
print(f"  num_mel_bins: {feature_extractor.num_mel_bins}")

print("\nTest forward pass...")
dummy_input = torch.randn(1, feature_extractor.max_length, feature_extractor.num_mel_bins)
print(f"Input shape: {dummy_input.shape}")
output = model(dummy_input)
print(f"Output shape: {output.last_hidden_state.shape}")
print("SUCCESS!")
