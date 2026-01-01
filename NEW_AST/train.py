"""
AST Training Script for Heart Sound Classification.

Usage:
    python train.py                      # Run with auto-detected profile
    python train.py --profile cpu        # Run on CPU
    python train.py --epochs 50          # Custom epochs
"""

import os
import sys
import argparse
import torch
import yaml
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ast_model import create_model
from model.dataset import create_dataloaders
from common.losses import create_loss_function
from common.trainer import Trainer


GPU_PROFILES = {
    'l4': {
        'batch_size': 8,
        'num_workers': 2,
        'learning_rate': 0.00005,
        'epochs': 50,
        'early_stopping_patience': 10,
        'gradient_clip': 1.0,
        'weight_decay': 0.01,
        'scheduler': 'cosine'
    },
    't4': {
        'batch_size': 4,
        'num_workers': 2,
        'learning_rate': 0.00005,
        'epochs': 50,
        'early_stopping_patience': 10,
        'gradient_clip': 1.0,
        'weight_decay': 0.01,
        'scheduler': 'cosine'
    },
    'low_mem': {
        'batch_size': 2,
        'num_workers': 0,
        'learning_rate': 0.00003,
        'epochs': 50,
        'early_stopping_patience': 10,
        'gradient_clip': 1.0,
        'weight_decay': 0.01,
        'scheduler': 'cosine'
    },
    'cpu': {
        'batch_size': 2,
        'num_workers': 0,
        'learning_rate': 0.00003,
        'epochs': 30,
        'early_stopping_patience': 8,
        'gradient_clip': 1.0,
        'weight_decay': 0.01,
        'scheduler': 'cosine'
    }
}

MODEL_CONFIG = {
    'model_type': 'pretrained',
    'pretrained_model': 'MIT/ast-finetuned-audioset-10-10-0.4593',
    'num_classes': 2,
    'freeze_encoder': False
}


def detect_device():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if 'L4' in gpu_name or gpu_memory >= 20:
            return 'cuda', 'l4'
        elif 'T4' in gpu_name or gpu_memory >= 12:
            return 'cuda', 't4'
        else:
            return 'cuda', 't4'
    else:
        print("No GPU detected, using CPU")
        return 'cpu', 'cpu'


def print_system_info():
    print("\n" + "="*60)
    print("AST HEART SOUND CLASSIFIER - TRAINING")
    print("="*60)
    
    import platform
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Memory: {props.total_memory / (1024**3):.2f} GB")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train AST Heart Classifier')
    parser.add_argument('--profile', type=str, choices=['l4', 't4', 'low_mem', 'cpu'],
                       help='Hardware profile (auto-detected if not specified)')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--freeze-encoder', action='store_true', help='Freeze AST encoder weights')
    parser.add_argument('--augment', action='store_true', default=True, help='Enable augmentation')
    args = parser.parse_args()
    
    print_system_info()
    
    device, auto_profile = detect_device()
    profile = args.profile or auto_profile
    config = GPU_PROFILES[profile].copy()
    
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'DATA')
    
    config['checkpoint_dir'] = os.path.join(script_dir, 'model', 'checkpoints')
    config.update(MODEL_CONFIG)
    
    if args.freeze_encoder:
        config['freeze_encoder'] = True
    
    print(f"Training Profile: {profile.upper()}")
    print(f"Device: {device}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Freeze Encoder: {config['freeze_encoder']}")
    print()
    
    train_csv = os.path.join(data_dir, 'cleaned_data_entries', 'train.csv')
    val_csv = os.path.join(data_dir, 'cleaned_data_entries', 'val.csv')
    train_dir = os.path.join(data_dir, 'cleaned_data', 'train')
    val_dir = os.path.join(data_dir, 'cleaned_data', 'val')
    
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augment_train=args.augment
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print()
    
    print("Creating AST model...")
    model = create_model(
        model_type=config['model_type'],
        pretrained_model=config['pretrained_model'],
        num_classes=config['num_classes'],
        freeze_encoder=config['freeze_encoder']
    )
    print()
    
    criterion = create_loss_function(
        loss_type='focal',
        pos_weight_heart=1.0,
        pos_weight_murmur=3.5
    )
    
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    print("Starting training...")
    print("="*60)
    
    start_time = datetime.now()
    history = trainer.train()
    end_time = datetime.now()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {end_time - start_time}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Best model saved to: {config['checkpoint_dir']}/best_model.pt")
    
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.yaml')
    with open(history_path, 'w') as f:
        yaml.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
