"""
Jivascope Training Script

Run this script to train the heart sound classification model.

Usage:
    python train.py                     # Interactive mode (detects system, asks for profile)
    python train.py --profile t4        # Use T4 profile directly
    python train.py --profile l4        # Use L4 profile directly
    python train.py --profile l40s      # Use L40S profile directly
    python train.py --epochs 30         # Train for 30 epochs
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import pandas as pd


GPU_PROFILES = {
    't4': {
        'name': 'NVIDIA T4',
        'memory_gb': 16,
        'batch_size': 16,
        'num_workers': 2,
        'accumulation_steps': 2,
        'description': 'Google Colab Free/Pro (T4 16GB)',
        'estimated_time_per_epoch': '3-4 min',
        'estimated_total_50_epochs': '~2.5-3 hours',
    },
    'l4': {
        'name': 'NVIDIA L4',
        'memory_gb': 24,
        'batch_size': 32,
        'num_workers': 4,
        'accumulation_steps': 1,
        'description': 'Cloud GPU (L4 24GB, 8 CPUs)',
        'estimated_time_per_epoch': '1.5-2 min',
        'estimated_total_50_epochs': '~1.5-2 hours',
    },
    'l40s': {
        'name': 'NVIDIA L40S',
        'memory_gb': 48,
        'batch_size': 64,
        'num_workers': 8,
        'accumulation_steps': 1,
        'description': 'High-end Cloud GPU (L40S 48GB, 16 CPUs)',
        'estimated_time_per_epoch': '45-60 sec',
        'estimated_total_50_epochs': '~40-50 min',
    },
    'cpu': {
        'name': 'CPU Only',
        'memory_gb': 0,
        'batch_size': 4,
        'num_workers': 0,
        'accumulation_steps': 4,
        'description': 'CPU training (slow, for testing only)',
        'estimated_time_per_epoch': '15-30 min',
        'estimated_total_50_epochs': '~12-24 hours',
    }
}


def get_system_info():
    """Detect and return system specifications."""
    import platform
    import psutil
    
    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=True),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        'ram_available_gb': round(psutil.virtual_memory().available / (1024**3), 1),
    }
    
    info['cuda_available'] = torch.cuda.is_available()
    
    if info['cuda_available']:
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        
        torch.cuda.empty_cache()
        info['gpu_memory_free_gb'] = round(
            (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3), 1
        )
    else:
        info['cuda_version'] = None
        info['gpu_count'] = 0
        info['gpu_name'] = None
        info['gpu_memory_total_gb'] = 0
        info['gpu_memory_free_gb'] = 0
    
    return info


def print_system_info(info):
    """Display system information in a formatted way."""
    print("\n" + "="*60)
    print("📊 SYSTEM SPECIFICATIONS")
    print("="*60)
    
    print("\n🖥️  CPU & Memory:")
    print(f"    Platform:     {info['platform']}")
    print(f"    Python:       {info['python_version']}")
    print(f"    CPU Cores:    {info['cpu_count']} logical ({info['cpu_count_physical']} physical)")
    print(f"    RAM Total:    {info['ram_total_gb']} GB")
    print(f"    RAM Free:     {info['ram_available_gb']} GB")
    
    print("\n🎮 GPU:")
    if info['cuda_available']:
        print(f"    CUDA:         ✅ Available (v{info['cuda_version']})")
        print(f"    GPU Count:    {info['gpu_count']}")
        print(f"    GPU Name:     {info['gpu_name']}")
        print(f"    GPU Memory:   {info['gpu_memory_total_gb']} GB total")
        print(f"    GPU Free:     {info['gpu_memory_free_gb']} GB available")
    else:
        print("    CUDA:         ❌ Not Available")
        print("    GPU:          None detected")
    
    print("="*60)


def detect_gpu_profile(info):
    """Attempt to auto-detect the best matching GPU profile."""
    if not info['cuda_available']:
        return 'cpu'
    
    gpu_mem = info['gpu_memory_total_gb']
    gpu_name = info['gpu_name'].lower() if info['gpu_name'] else ''
    
    if 'l40s' in gpu_name or gpu_mem >= 45:
        return 'l40s'
    elif 'l4' in gpu_name or (gpu_mem >= 22 and gpu_mem < 45):
        return 'l4'
    elif 't4' in gpu_name or (gpu_mem >= 14 and gpu_mem < 22):
        return 't4'
    elif gpu_mem > 0:
        return 't4'
    
    return 'cpu'


def print_profiles():
    """Display available GPU profiles."""
    print("\n📋 Available Training Profiles:")
    print("-" * 60)
    
    t4 = GPU_PROFILES['t4']
    print(f"\n  [1] T4 Profile")
    print(f"      GPU Memory:    {t4['memory_gb']} GB")
    print(f"      Batch Size:    {t4['batch_size']}")
    print(f"      Est. Time/Epoch: {t4['estimated_time_per_epoch']}")
    print(f"      Est. Total (50): {t4['estimated_total_50_epochs']}")
    print(f"      Best for: Google Colab Free/Pro")
    
    l4 = GPU_PROFILES['l4']
    print(f"\n  [2] L4 Profile")
    print(f"      GPU Memory:    {l4['memory_gb']} GB")
    print(f"      Batch Size:    {l4['batch_size']}")
    print(f"      Est. Time/Epoch: {l4['estimated_time_per_epoch']}")
    print(f"      Est. Total (50): {l4['estimated_total_50_epochs']}")
    print(f"      Best for: Cloud instances with L4 GPU, 8 CPUs")
    
    l40s = GPU_PROFILES['l40s']
    print(f"\n  [3] L40S Profile")
    print(f"      GPU Memory:    {l40s['memory_gb']} GB")
    print(f"      Batch Size:    {l40s['batch_size']}")
    print(f"      Est. Time/Epoch: {l40s['estimated_time_per_epoch']}")
    print(f"      Est. Total (50): {l40s['estimated_total_50_epochs']}")
    print(f"      Best for: High-end cloud (L40S, 16 CPUs)")
    
    cpu = GPU_PROFILES['cpu']
    print(f"\n  [4] CPU Only")
    print(f"      Batch Size:    {cpu['batch_size']}")
    print(f"      Est. Time/Epoch: {cpu['estimated_time_per_epoch']}")
    print(f"      Est. Total (50): {cpu['estimated_total_50_epochs']}")
    print(f"      Best for: Testing only (very slow)")
    
    print("-" * 60)


def get_user_profile_choice(detected_profile):
    """Ask user to select a training profile."""
    print_profiles()
    
    print(f"\n💡 Based on your system, recommended profile: {detected_profile.upper()}")
    print("\nEnter your choice (1-4) or press Enter for recommended: ", end="")
    
    try:
        choice = input().strip()
        
        if choice == '' or choice == '0':
            return detected_profile
        elif choice == '1':
            return 't4'
        elif choice == '2':
            return 'l4'
        elif choice == '3':
            return 'l40s'
        elif choice == '4':
            return 'cpu'
        else:
            print(f"Invalid choice '{choice}', using recommended: {detected_profile}")
            return detected_profile
    except (EOFError, KeyboardInterrupt):
        print(f"\nUsing recommended profile: {detected_profile}")
        return detected_profile


def apply_profile(config, profile_name):
    """Apply GPU profile settings to config."""
    profile = GPU_PROFILES[profile_name]
    
    config.training.batch_size = profile['batch_size']
    config.num_workers = profile['num_workers']
    
    if profile_name == 'cpu':
        config.device = 'cpu'
    else:
        config.device = 'cuda'
    
    print(f"\n✅ Applied '{profile_name.upper()}' profile:")
    print(f"   Batch Size: {profile['batch_size']}")
    print(f"   Num Workers: {profile['num_workers']}")
    print(f"   Device: {config.device}")
    
    return config, profile


def parse_args():
    parser = argparse.ArgumentParser(description='Train Jivascope Heart Sound Classifier')
    
    parser.add_argument('--profile', type=str, default=None, 
                        choices=['t4', 'l4', 'l40s', 'cpu'],
                        help='GPU profile to use (skips interactive selection)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from profile')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--backbone', type=str, default=None,
                        choices=['efficientnet_b0', 'resnet18', 'resnet34'],
                        help='Model backbone (default: efficientnet_b0)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--skip_check', action='store_true',
                        help='Skip system check (not recommended)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("🫀 JIVASCOPE - Heart Sound Classification Training")
    print("="*60)
    
    if not args.skip_check:
        print("\n🔍 Detecting system specifications...")
        try:
            import psutil
        except ImportError:
            print("Installing psutil for system detection...")
            os.system('pip install psutil -q')
            import psutil
        
        sys_info = get_system_info()
        print_system_info(sys_info)
        
        detected_profile = detect_gpu_profile(sys_info)
        
        if args.profile:
            selected_profile = args.profile
            print(f"\n📌 Using specified profile: {selected_profile.upper()}")
        else:
            selected_profile = get_user_profile_choice(detected_profile)
    else:
        selected_profile = args.profile or 't4'
        print(f"\n⚠️ Skipping system check, using profile: {selected_profile.upper()}")
    
    from src.models import create_model
    from src.data.dataset import create_dataloaders, get_class_weights
    from src.training import create_trainer
    from src.utils.config import get_default_config, print_config
    
    config = get_default_config()
    
    config, profile = apply_profile(config, selected_profile)
    
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.backbone:
        config.model.backbone = args.backbone
    if args.no_pretrained:
        config.model.pretrained = False
    
    if config.device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️ CUDA not available, falling back to CPU")
        config.device = 'cpu'
        config.training.batch_size = 4
        config.num_workers = 0
    
    print("\n" + "-"*60)
    print("FINAL CONFIGURATION")
    print("-"*60)
    print_config(config)
    
    print("\n📂 Loading data...")
    
    train_csv = config.paths.cleaned_data_dir / 'train.csv'
    val_csv = config.paths.cleaned_data_dir / 'val.csv'
    test_csv = config.paths.cleaned_data_dir / 'test.csv'
    
    if not train_csv.exists():
        print(f"\n❌ Error: Training data not found at {train_csv}")
        print("\nPlease ensure the cleaned_data folder exists with:")
        print("  - train.csv")
        print("  - val.csv")
        print("  - test.csv")
        print("\nRun 'python src/data/clean_data.py' if you haven't created the splits yet.")
        sys.exit(1)
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"  Train: {len(train_df)} patients")
    print(f"  Val:   {len(val_df)} patients")
    print(f"  Test:  {len(test_df)} patients")
    
    print("\n📊 Computing class weights for imbalanced data...")
    class_weights = get_class_weights(train_df)
    print(f"  Murmur weights:  {[round(w, 3) for w in class_weights['murmur_weights'].tolist()]}")
    print(f"  Outcome weights: {[round(w, 3) for w in class_weights['outcome_weights'].tolist()]}")
    
    print(f"\n🔄 Creating data loaders (batch_size={config.training.batch_size}, workers={config.num_workers})...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, 
        val_df, 
        test_df,
        audio_dir=str(config.paths.audio_dir),
        batch_size=config.training.batch_size,
        num_workers=config.num_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    print("\n🤖 Creating model...")
    model = create_model(
        backbone_name=config.model.backbone,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
        device=config.device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone:   {config.model.backbone}")
    print(f"  Pretrained: {config.model.pretrained}")
    print(f"  Parameters: {trainable_params:,} trainable / {total_params:,} total")
    
    if config.device == 'cuda':
        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        mem_reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"  GPU Memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
    
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
    print(f"🚀 STARTING TRAINING ({config.training.num_epochs} epochs)")
    print(f"   Profile:    {selected_profile.upper()}")
    print(f"   Device:     {config.device}")
    print(f"   Batch Size: {config.training.batch_size}")
    print(f"   Est. Time:  {profile['estimated_time_per_epoch']} per epoch")
    print(f"   Est. Total: {profile['estimated_total_50_epochs']} (for 50 epochs)")
    print("="*60 + "\n")
    
    history = trainer.fit()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Checkpoints saved to: {config.paths.checkpoint_dir}")
    print(f"   - best_model.pt  (best validation loss)")
    print(f"   - final_model.pt (last epoch)")
    print(f"\n📊 TensorBoard logs: {config.paths.log_dir}")
    print("\nTo view training curves, run:")
    print(f"   tensorboard --logdir {config.paths.log_dir}")
    
    return history


if __name__ == "__main__":
    main()
