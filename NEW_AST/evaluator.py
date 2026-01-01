"""
AST Model Evaluator for Heart Sound Classification.

Evaluates the trained AST model on test dataset, computing comprehensive
metrics including accuracy, sensitivity, specificity, precision, and F1 score.

Usage:
    python evaluator.py
    python evaluator.py --samples 100
    python evaluator.py --model path/to/model.pt --samples 50
"""

import os
import sys
import argparse
import random
import torch
import numpy as np
import pandas as pd
import librosa
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ast_model import create_model
from model.dataset import extract_ast_features, AST_SAMPLE_RATE, AST_TARGET_DURATION
from common.audio import load_audio, apply_bandpass_filter, normalize_audio, pad_or_truncate


class ASTEvaluator:
    """Evaluator class for AST heart sound classification model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        config = checkpoint.get('config', {})
        self.model = create_model(
            model_type=config.get('model_type', 'pretrained'),
            pretrained_model=config.get('pretrained_model', 'MIT/ast-finetuned-audioset-10-10-0.4593'),
            num_classes=config.get('num_classes', 2)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded successfully on {device}")
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        audio = load_audio(audio_path, target_sr=4000)
        audio = apply_bandpass_filter(audio, sr=4000)
        audio = normalize_audio(audio)
        
        target_length_4k = int(AST_TARGET_DURATION * 4000)
        audio = pad_or_truncate(audio, target_length_4k)
        
        audio_16k = librosa.resample(audio, orig_sr=4000, target_sr=AST_SAMPLE_RATE)
        features = extract_ast_features(audio_16k, sr=AST_SAMPLE_RATE)
        
        return features.unsqueeze(0)
    
    @torch.no_grad()
    def predict_single(self, audio_path: str) -> Dict:
        """Run prediction on single audio file."""
        features = self.preprocess_audio(audio_path)
        features = features.to(self.device)
        
        logits = self.model(features)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        heart_sound_present = bool(probs[0] > 0.5)
        murmur_present = bool(probs[1] > 0.5) if heart_sound_present else False
        
        return {
            "heart_sound_present": heart_sound_present,
            "murmur_present": murmur_present,
            "heart_sound_prob": float(probs[0]),
            "murmur_prob": float(probs[1])
        }
    
    def evaluate_dataset(
        self,
        test_csv: str,
        test_dir: str,
        num_samples: int = 50,
        seed: int = None
    ) -> Dict:
        """
        Evaluate model on test dataset.
        
        Args:
            test_csv: Path to test CSV file
            test_dir: Path to test audio directory
            num_samples: Number of samples to evaluate (randomly selected)
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary containing evaluation metrics and detailed results
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        df = pd.read_csv(test_csv)
        
        valid_samples = []
        for idx, row in df.iterrows():
            patient_id = str(row['Patient ID'])
            outcome_label = int(row['Outcome_Label'])
            wav_path = os.path.join(test_dir, f"{patient_id}.wav")
            
            if os.path.exists(wav_path):
                valid_samples.append({
                    'patient_id': patient_id,
                    'file_path': wav_path,
                    'ground_truth': outcome_label
                })
        
        print(f"Found {len(valid_samples)} valid samples in test set")
        
        if num_samples > len(valid_samples):
            print(f"Requested {num_samples} samples but only {len(valid_samples)} available. Using all.")
            selected_samples = valid_samples
        else:
            selected_samples = random.sample(valid_samples, num_samples)
        
        print(f"Evaluating on {len(selected_samples)} samples...")
        print("-" * 60)
        
        results = []
        y_true = []
        y_pred = []
        y_probs = []
        
        for i, sample in enumerate(selected_samples):
            try:
                prediction = self.predict_single(sample['file_path'])
                
                predicted_label = 1 if prediction['murmur_present'] else 0
                ground_truth = sample['ground_truth']
                
                y_true.append(ground_truth)
                y_pred.append(predicted_label)
                y_probs.append(prediction['murmur_prob'])
                
                is_correct = predicted_label == ground_truth
                
                results.append({
                    'patient_id': sample['patient_id'],
                    'ground_truth': ground_truth,
                    'predicted': predicted_label,
                    'murmur_prob': prediction['murmur_prob'],
                    'correct': is_correct
                })
                
                status = "✓" if is_correct else "✗"
                gt_label = "Murmur" if ground_truth == 1 else "Normal"
                pred_label = "Murmur" if predicted_label == 1 else "Normal"
                
                print(f"[{i+1:3d}/{len(selected_samples)}] {status} {sample['patient_id']}: "
                      f"GT={gt_label:7s}, Pred={pred_label:7s}, Prob={prediction['murmur_prob']:.3f}")
                      
            except Exception as e:
                print(f"[{i+1:3d}/{len(selected_samples)}] ERROR {sample['patient_id']}: {str(e)}")
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        metrics = self._compute_metrics(y_true, y_pred, y_probs)
        
        return {
            'metrics': metrics,
            'results': results,
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_probs': y_probs.tolist()
        }
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray
    ) -> Dict:
        """Compute comprehensive classification metrics."""
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        balanced_accuracy = (sensitivity + specificity) / 2
        
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'f1_score': f1,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc,
            'confusion_matrix': {
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            },
            'class_distribution': {
                'total_samples': len(y_true),
                'positive_samples': int(np.sum(y_true == 1)),
                'negative_samples': int(np.sum(y_true == 0))
            }
        }


def print_report(eval_results: Dict):
    """Print formatted evaluation report."""
    metrics = eval_results['metrics']
    cm = metrics['confusion_matrix']
    dist = metrics['class_distribution']
    
    print("\n" + "=" * 70)
    print("                    AST MODEL EVALUATION REPORT")
    print("=" * 70)
    print(f"  Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    
    print("\n  DATASET SUMMARY")
    print("-" * 70)
    print(f"  Total Samples Evaluated: {dist['total_samples']}")
    print(f"  Positive Samples (Murmur): {dist['positive_samples']} ({100*dist['positive_samples']/dist['total_samples']:.1f}%)")
    print(f"  Negative Samples (Normal): {dist['negative_samples']} ({100*dist['negative_samples']/dist['total_samples']:.1f}%)")
    
    print("\n  PERFORMANCE METRICS")
    print("-" * 70)
    print(f"  {'Metric':<30} {'Value':>15}")
    print("  " + "-" * 46)
    print(f"  {'Accuracy':<30} {metrics['accuracy']*100:>14.2f}%")
    print(f"  {'Balanced Accuracy':<30} {metrics['balanced_accuracy']*100:>14.2f}%")
    print(f"  {'Sensitivity (Recall/TPR)':<30} {metrics['sensitivity']*100:>14.2f}%")
    print(f"  {'Specificity (TNR)':<30} {metrics['specificity']*100:>14.2f}%")
    print(f"  {'Precision (PPV)':<30} {metrics['precision']*100:>14.2f}%")
    print(f"  {'Negative Predictive Value':<30} {metrics['npv']*100:>14.2f}%")
    print(f"  {'F1 Score':<30} {metrics['f1_score']*100:>14.2f}%")
    print(f"  {'Matthews Correlation Coeff.':<30} {metrics['mcc']:>15.4f}")
    
    print("\n  CONFUSION MATRIX")
    print("-" * 70)
    print("                          Predicted")
    print("                    Normal      Murmur")
    print(f"  Actual  Normal    {cm['true_negatives']:>6}      {cm['false_positives']:>6}")
    print(f"          Murmur    {cm['false_negatives']:>6}      {cm['true_positives']:>6}")
    
    print("\n  SENSITIVITY ANALYSIS")
    print("-" * 70)
    print("  Sensitivity measures how well the model detects positive cases (murmurs).")
    print(f"  - Current Sensitivity: {metrics['sensitivity']*100:.2f}%")
    if metrics['sensitivity'] >= 0.90:
        print("  - Rating: EXCELLENT - Model catches most murmur cases")
    elif metrics['sensitivity'] >= 0.80:
        print("  - Rating: GOOD - Model misses some murmur cases")
    elif metrics['sensitivity'] >= 0.70:
        print("  - Rating: MODERATE - Model misses notable murmur cases")
    else:
        print("  - Rating: NEEDS IMPROVEMENT - Model misses too many murmur cases")
    
    print("\n  Specificity measures how well the model identifies normal cases.")
    print(f"  - Current Specificity: {metrics['specificity']*100:.2f}%")
    if metrics['specificity'] >= 0.90:
        print("  - Rating: EXCELLENT - Model rarely flags normal as murmur")
    elif metrics['specificity'] >= 0.80:
        print("  - Rating: GOOD - Model occasionally flags normal as murmur")
    elif metrics['specificity'] >= 0.70:
        print("  - Rating: MODERATE - Model somewhat over-diagnoses")
    else:
        print("  - Rating: NEEDS IMPROVEMENT - Model over-diagnoses frequently")
    
    print("\n" + "=" * 70)
    print("                         END OF REPORT")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate AST Heart Sound Model')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None)
    parser.add_argument('--all', action='store_true', help='Evaluate on all test samples')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'DATA')
    
    if args.model is None:
        args.model = os.path.join(script_dir, 'model', 'checkpoints', 'best_acc_model.pt')
    
    test_csv = os.path.join(data_dir, 'cleaned_data_entries', 'test.csv')
    test_dir = os.path.join(data_dir, 'cleaned_data', 'test')
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(test_csv):
        print(f"Error: Test CSV not found: {test_csv}")
        sys.exit(1)
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    evaluator = ASTEvaluator(args.model, args.device)
    
    num_samples = 999999 if args.all else args.samples
    
    eval_results = evaluator.evaluate_dataset(
        test_csv=test_csv,
        test_dir=test_dir,
        num_samples=num_samples,
        seed=args.seed
    )
    
    print_report(eval_results)
    
    return eval_results


if __name__ == "__main__":
    main()
