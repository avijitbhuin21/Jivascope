"""
Debug script to test different AST model loading methods.
"""

import os
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

print("=" * 60)
print("AST MODEL LOADING DEBUG")
print("=" * 60)

print("\n[Test 1] Basic PyTorch import...")
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
gc.collect()

print("\n[Test 2] Import transformers without loading model...")
try:
    from transformers import ASTConfig
    print("  ASTConfig imported successfully")
except Exception as e:
    print(f"  FAILED: {e}")
gc.collect()

print("\n[Test 3] Create AST model from scratch (no pretrained weights)...")
try:
    from transformers import ASTModel, ASTConfig
    config = ASTConfig(
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        max_length=1024,
        frequency_size=128
    )
    model = ASTModel(config)
    print(f"  Model created: {sum(p.numel() for p in model.parameters()):,} params")
    del model
    gc.collect()
except Exception as e:
    print(f"  FAILED: {e}")

print("\n[Test 4] Load pretrained with safetensors only...")
try:
    from transformers import ASTModel
    model = ASTModel.from_pretrained(
        'MIT/ast-finetuned-audioset-10-10-0.4593',
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    del model
    gc.collect()
except Exception as e:
    print(f"  FAILED: {e}")

print("\n[Test 5] Load with torch_dtype float16...")
try:
    from transformers import ASTModel
    model = ASTModel.from_pretrained(
        'MIT/ast-finetuned-audioset-10-10-0.4593',
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    del model
    gc.collect()
except Exception as e:
    print(f"  FAILED: {e}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
