"""
Jivascope Training Script

Run this script to train the heart sound classification model.

Usage:
    python train.py                     # Train with default settings
    python train.py --epochs 30         # Train for 30 epochs
    python train.py --batch_size 8      # Use smaller batch size (for limited GPU memory)
    python train.py --device cpu        # Train on CPU (slow, for testing only)
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import pandas as pd

from src.models import create_model
from src.data.dataset import create_dataloaders, get_class_weights
from src.training import create_trainer
from src.utils.config import get_default_config, print_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train Jivascope Heart Sound Classifier')
    
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Device to train on (default: cuda if available)')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['efficientnet_b0', 'resnet18', 'resnet34'],
                        help='Model backbone (default: efficientnet_b0)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = get_default_config()
    
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.device:
        config.device = args.device
    if args.backbone:
        config.model.backbone = args.backbone
    if args.no_pretrained:
        config.model.pretrained = False
    
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        config.device = 'cpu'
    
    print("\n" + "="*60)
    print("🫀 JIVASCOPE - Heart Sound Classification")
    print("="*60)
    print_config(config)
    
    print("\n📂 Loading data...")
    
    train_csv = config.paths.cleaned_data_dir / 'train.csv'
    val_csv = config.paths.cleaned_data_dir / 'val.csv'
    test_csv = config.paths.cleaned_data_dir / 'test.csv'
    
    if not train_csv.exists():
        print(f"❌ Error: Training data not found at {train_csv}")
        print("Please ensure the cleaned_data folder exists with train.csv, val.csv, test.csv")
        sys.exit(1)
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"  Train samples: {len(train_df)} patients")
    print(f"  Val samples: {len(val_df)} patients")
    print(f"  Test samples: {len(test_df)} patients")
    
    print("\n📊 Computing class weights...")
    class_weights = get_class_weights(train_df)
    print(f"  Murmur weights: {class_weights['murmur_weights'].tolist()}")
    print(f"  Outcome weights: {class_weights['outcome_weights'].tolist()}")
    
    print("\n🔄 Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, 
        val_df, 
        test_df,
        audio_dir=str(config.paths.audio_dir),
        batch_size=config.training.batch_size,
        num_workers=config.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    print("\n🤖 Creating model...")
    model = create_model(
        backbone_name=config.model.backbone,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        device=config.device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone: {config.model.backbone}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n🏋️ Initializing trainer...")
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_weights=class_weights
    )
    
    if args.resume:
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("\n" + "="*60)
    print("🚀 Starting training...")
    print("="*60)
    
    history = trainer.fit()
    
    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)
    print(f"\n📁 Checkpoints saved to: {config.paths.checkpoint_dir}")
    print(f"📊 TensorBoard logs: {config.paths.log_dir}")
    print("\nTo view training curves, run:")
    print(f"  tensorboard --logdir {config.paths.log_dir}")
    
    return history


if __name__ == "__main__":
    main()
