"""
Jivascope Model Validation Script

Compare the performance of trained models on the test dataset.

Usage:
    python validate.py                              # Compare all models in checkpoints/
    python validate.py --model final_model.pt       # Test single model
    python validate.py --runs 10                    # Run 10 times instead of 5
    python validate.py --show-predictions 20        # Show 20 sample predictions
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from src.models import HeartSoundClassifier, create_model
from src.data.dataset import HeartSoundDataset
from src.utils.config import get_default_config


MURMUR_LABELS = ['Absent', 'Present']
OUTCOME_LABELS = ['Normal', 'Abnormal']


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Tuple[HeartSoundClassifier, Dict]:
    """
    Load a trained model from checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'backbone_name' in checkpoint:
        backbone_name = checkpoint['backbone_name']
    elif 'config' in checkpoint and hasattr(checkpoint['config'], 'model'):
        backbone_name = checkpoint['config'].model.backbone
    else:
        file_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
        backbone_name = 'resnet18' if file_size_mb > 100 else 'efficientnet_b0'
        print(f"  ⚠️ Backbone not in checkpoint, inferring '{backbone_name}' from file size ({file_size_mb:.1f}MB)")
    
    model = create_model(
        backbone_name=backbone_name,
        pretrained=False,
        device=device
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    info = {
        'backbone': backbone_name,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_val_loss': checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'N/A')),
    }
    
    return model, info


def evaluate_model(
    model: HeartSoundClassifier,
    test_dataset: HeartSoundDataset,
    device: str = 'cpu',
    batch_size: int = 16
) -> Dict[str, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run on
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary with accuracy metrics
    """
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    model.eval()
    
    all_murmur_preds = []
    all_murmur_labels = []
    all_outcome_preds = []
    all_outcome_labels = []
    
    with torch.no_grad():
        for spectrograms, murmur_labels, outcome_labels, _ in test_loader:
            spectrograms = spectrograms.to(device)
            
            predictions = model.get_predictions(spectrograms)
            
            all_murmur_preds.extend(predictions['murmur'].cpu().numpy())
            all_murmur_labels.extend(murmur_labels.numpy())
            all_outcome_preds.extend(predictions['outcome'].cpu().numpy())
            all_outcome_labels.extend(outcome_labels.numpy())
    
    all_murmur_preds = np.array(all_murmur_preds)
    all_murmur_labels = np.array(all_murmur_labels)
    all_outcome_preds = np.array(all_outcome_preds)
    all_outcome_labels = np.array(all_outcome_labels)
    
    murmur_acc = (all_murmur_preds == all_murmur_labels).mean() * 100
    outcome_acc = (all_outcome_preds == all_outcome_labels).mean() * 100
    combined_acc = (murmur_acc + outcome_acc) / 2
    
    return {
        'murmur_accuracy': murmur_acc,
        'outcome_accuracy': outcome_acc,
        'combined_accuracy': combined_acc,
        'total_samples': len(all_murmur_labels)
    }


def get_sample_predictions(
    model: HeartSoundClassifier,
    test_dataset: HeartSoundDataset,
    device: str = 'cpu',
    num_samples: int = 10
) -> List[Dict]:
    """
    Get sample predictions with probabilities for display.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run on
        num_samples: Number of samples to show
        
    Returns:
        List of prediction dictionaries
    """
    model.eval()
    predictions_list = []
    
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            spectrogram, murmur_label, outcome_label, patient_id = test_dataset[idx]
            spectrogram = spectrogram.unsqueeze(0).to(device)
            
            probs = model.get_probabilities(spectrogram)
            preds = model.get_predictions(spectrogram)
            
            murmur_prob = probs['murmur'][0].cpu().numpy()
            outcome_prob = probs['outcome'][0].cpu().numpy()
            murmur_pred = preds['murmur'][0].cpu().item()
            outcome_pred = preds['outcome'][0].cpu().item()
            
            predictions_list.append({
                'patient_id': patient_id,
                'murmur_true': MURMUR_LABELS[murmur_label],
                'murmur_pred': MURMUR_LABELS[murmur_pred],
                'murmur_confidence': murmur_prob[murmur_pred] * 100,
                'murmur_correct': murmur_pred == murmur_label,
                'outcome_true': OUTCOME_LABELS[outcome_label],
                'outcome_pred': OUTCOME_LABELS[outcome_pred],
                'outcome_confidence': outcome_prob[outcome_pred] * 100,
                'outcome_correct': outcome_pred == outcome_label,
            })
    
    return predictions_list


def run_validation(
    checkpoint_path: str,
    test_dataset: HeartSoundDataset,
    num_runs: int = 5,
    show_predictions: int = 10,
    device: str = 'cpu'
) -> Dict:
    """
    Run full validation for a single model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_dataset: Test dataset
        num_runs: Number of evaluation runs
        show_predictions: Number of sample predictions to show
        device: Device to run on
        
    Returns:
        Results dictionary
    """
    model_name = Path(checkpoint_path).name
    
    print(f"\n{'='*60}")
    print(f"🔬 Evaluating: {model_name}")
    print(f"{'='*60}")
    
    print(f"\n📥 Loading model...")
    model, info = load_model_from_checkpoint(checkpoint_path, device)
    print(f"   Backbone:      {info['backbone']}")
    print(f"   Epoch:         {info['epoch']}")
    if isinstance(info['best_val_loss'], float):
        print(f"   Best Val Loss: {info['best_val_loss']:.4f}")
    
    run_results = []
    
    print(f"\n📊 Running {num_runs} evaluation(s)...")
    for run in range(num_runs):
        metrics = evaluate_model(model, test_dataset, device)
        run_results.append(metrics)
        
        print(f"   Run {run + 1}/{num_runs}: "
              f"Murmur={metrics['murmur_accuracy']:.2f}%, "
              f"Outcome={metrics['outcome_accuracy']:.2f}%")
    
    avg_murmur = np.mean([r['murmur_accuracy'] for r in run_results])
    std_murmur = np.std([r['murmur_accuracy'] for r in run_results])
    avg_outcome = np.mean([r['outcome_accuracy'] for r in run_results])
    std_outcome = np.std([r['outcome_accuracy'] for r in run_results])
    avg_combined = np.mean([r['combined_accuracy'] for r in run_results])
    std_combined = np.std([r['combined_accuracy'] for r in run_results])
    
    print(f"\n📈 Average Results ({num_runs} runs):")
    print(f"   Murmur Accuracy:  {avg_murmur:.2f}% ± {std_murmur:.2f}%")
    print(f"   Outcome Accuracy: {avg_outcome:.2f}% ± {std_outcome:.2f}%")
    print(f"   Combined:         {avg_combined:.2f}% ± {std_combined:.2f}%")
    
    if show_predictions > 0:
        print(f"\n🎯 Sample Predictions ({show_predictions} samples):")
        print("-" * 60)
        
        sample_preds = get_sample_predictions(model, test_dataset, device, show_predictions)
        
        for pred in sample_preds:
            murmur_mark = "✓" if pred['murmur_correct'] else "✗"
            outcome_mark = "✓" if pred['outcome_correct'] else "✗"
            
            print(f"\n   {pred['patient_id']}:")
            print(f"     Murmur:  {pred['murmur_pred']:7s} ({pred['murmur_confidence']:5.1f}%) "
                  f"[True: {pred['murmur_true']}] {murmur_mark}")
            print(f"     Outcome: {pred['outcome_pred']:7s} ({pred['outcome_confidence']:5.1f}%) "
                  f"[True: {pred['outcome_true']}] {outcome_mark}")
    
    return {
        'model_name': model_name,
        'backbone': info['backbone'],
        'avg_murmur': avg_murmur,
        'std_murmur': std_murmur,
        'avg_outcome': avg_outcome,
        'std_outcome': std_outcome,
        'avg_combined': avg_combined,
        'std_combined': std_combined,
        'total_samples': run_results[0]['total_samples']
    }


def print_comparison_table(results: List[Dict]) -> None:
    """Print comparison table for all evaluated models."""
    print(f"\n{'='*60}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*60}\n")
    
    header = f"{'Model':<35} {'Backbone':<15} {'Murmur':<12} {'Outcome':<12} {'Combined':<12}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        model_short = r['model_name'][:33] if len(r['model_name']) > 33 else r['model_name']
        print(f"{model_short:<35} {r['backbone']:<15} "
              f"{r['avg_murmur']:>5.2f}%      {r['avg_outcome']:>5.2f}%      {r['avg_combined']:>5.2f}%")
    
    if len(results) > 1:
        best = max(results, key=lambda x: x['avg_combined'])
        second = sorted(results, key=lambda x: x['avg_combined'], reverse=True)[1]
        diff = best['avg_combined'] - second['avg_combined']
        
        print(f"\n🏆 Winner: {best['model_name']} ({best['backbone']}) - +{diff:.2f}% combined accuracy")


def parse_args():
    parser = argparse.ArgumentParser(description='Validate Jivascope Heart Sound Models')
    
    parser.add_argument('--model', type=str, default=None,
                        help='Path to specific model checkpoint (default: all in checkpoints/)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of evaluation runs per model (default: 5)')
    parser.add_argument('--show-predictions', type=int, default=10,
                        help='Number of sample predictions to display (default: 10)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation (default: 16)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("🫀 JIVASCOPE - Model Validation")
    print("="*60)
    
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    config = get_default_config()
    
    print("\n📂 Loading test data...")
    test_csv = config.paths.cleaned_data_dir / 'test.csv'
    
    if not test_csv.exists():
        print(f"\n❌ Error: Test data not found at {test_csv}")
        print("Please ensure cleaned_data/test.csv exists.")
        sys.exit(1)
    
    test_df = pd.read_csv(test_csv)
    print(f"   Patients: {len(test_df)}")
    
    test_dataset = HeartSoundDataset(
        df=test_df,
        audio_dir=str(config.paths.audio_dir),
        is_training=False
    )
    print(f"   Samples:  {len(test_dataset)}")
    
    if args.model:
        checkpoint_paths = [args.model]
    else:
        checkpoint_dir = config.paths.checkpoint_dir
        checkpoint_paths = list(checkpoint_dir.glob('*.pt'))
        
        if not checkpoint_paths:
            print(f"\n❌ Error: No checkpoints found in {checkpoint_dir}")
            sys.exit(1)
        
        print(f"\n📁 Found {len(checkpoint_paths)} checkpoint(s) in {checkpoint_dir}")
        for p in checkpoint_paths:
            print(f"   - {p.name}")
    
    all_results = []
    
    for checkpoint_path in checkpoint_paths:
        result = run_validation(
            checkpoint_path=str(checkpoint_path),
            test_dataset=test_dataset,
            num_runs=args.runs,
            show_predictions=args.show_predictions,
            device=device
        )
        all_results.append(result)
    
    if len(all_results) > 1:
        print_comparison_table(all_results)
    
    print("\n" + "="*60)
    print("✅ Validation Complete!")
    print("="*60 + "\n")
    
    return all_results


if __name__ == "__main__":
    main()
