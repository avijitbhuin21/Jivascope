"""
Model Comparison Script - AST vs LightCardiacNet.

Compares both heart murmur detection models on the same test samples
and generates a comprehensive comparison report.

Usage:
    python compare.py
    python compare.py --samples 50 --seed 42
    python compare.py --all
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
from scipy.signal import butter, filtfilt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

AST_SAMPLE_RATE = 16000
AST_TARGET_DURATION = 10.0
AST_N_MELS = 128
AST_N_FFT = 400
AST_HOP_LENGTH = 160
AST_MAX_LENGTH = 1024


def extract_ast_features(
    audio: np.ndarray,
    sr: int = AST_SAMPLE_RATE,
    n_mels: int = AST_N_MELS,
    n_fft: int = AST_N_FFT,
    hop_length: int = AST_HOP_LENGTH,
    max_length: int = AST_MAX_LENGTH
) -> torch.Tensor:
    """Extract log-mel spectrogram features for AST model."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=sr // 2
    )
    
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    log_mel = log_mel.T
    
    current_length = log_mel.shape[0]
    if current_length < max_length:
        padding = np.zeros((max_length - current_length, n_mels))
        log_mel = np.vstack([log_mel, padding])
    elif current_length > max_length:
        log_mel = log_mel[:max_length, :]
    
    return torch.tensor(log_mel, dtype=torch.float32)

LCN_SAMPLE_RATE = 4000
LCN_TARGET_DURATION = 10.0
LCN_N_MFCC = 13
LCN_N_FFT = 256
LCN_HOP_LENGTH = 64


def load_audio(file_path: str, target_sr: int = 4000) -> np.ndarray:
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio


def apply_bandpass_filter(
    audio: np.ndarray, 
    sr: int = 4000, 
    low_freq: float = 25.0, 
    high_freq: float = 400.0
) -> np.ndarray:
    nyquist = sr / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)
    
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)
    
    return filtered.astype(np.float32)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        audio = audio / rms
    return audio


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    current_length = len(audio)
    
    if current_length > target_length:
        start = (current_length - target_length) // 2
        return audio[start:start + target_length]
    elif current_length < target_length:
        pad_total = target_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant', constant_values=0)
    
    return audio


def extract_lcn_features(audio: np.ndarray, sr: int = LCN_SAMPLE_RATE) -> torch.Tensor:
    """Extract MFCC features with deltas for LightCardiacNet."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=LCN_N_MFCC,
        n_fft=LCN_N_FFT,
        hop_length=LCN_HOP_LENGTH
    )
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    features = np.vstack([mfcc, delta, delta2])
    features = features.T
    
    return torch.tensor(features, dtype=torch.float32)


class ASTModelWrapper:
    """Wrapper for AST model."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        
        print(f"Loading AST model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        config = checkpoint.get('config', {})
        
        ast_path = os.path.join(PROJECT_ROOT, 'NEW_AST')
        if ast_path not in sys.path:
            sys.path.insert(0, ast_path)
        
        from model.ast_model import create_model as create_ast_model
        
        self.model = create_ast_model(
            model_type=config.get('model_type', 'pretrained'),
            pretrained_model=config.get('pretrained_model', 'MIT/ast-finetuned-audioset-10-10-0.4593'),
            num_classes=config.get('num_classes', 2)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print("AST model loaded successfully")
    
    def preprocess(self, audio_path: str) -> torch.Tensor:
        audio = load_audio(audio_path, target_sr=4000)
        audio = apply_bandpass_filter(audio, sr=4000)
        audio = normalize_audio(audio)
        
        target_length_4k = int(AST_TARGET_DURATION * 4000)
        audio = pad_or_truncate(audio, target_length_4k)
        
        audio_16k = librosa.resample(audio, orig_sr=4000, target_sr=AST_SAMPLE_RATE)
        features = extract_ast_features(audio_16k, sr=AST_SAMPLE_RATE)
        
        return features.unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict:
        features = self.preprocess(audio_path)
        features = features.to(self.device)
        
        logits = self.model(features)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        heart_sound_present = bool(probs[0] > 0.5)
        murmur_present = bool(probs[1] > 0.5) if heart_sound_present else False
        
        return {
            "predicted": 1 if murmur_present else 0,
            "murmur_prob": float(probs[1]),
            "heart_sound_prob": float(probs[0])
        }


class LightCardiacNetWrapper:
    """Wrapper for LightCardiacNet model."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        
        print(f"Loading LightCardiacNet model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        config = checkpoint.get('config', {})
        
        lcn_path = os.path.join(PROJECT_ROOT, 'backup', 'OLD_1_lightCardiacNet')
        ast_path = os.path.join(PROJECT_ROOT, 'NEW_AST')
        
        if ast_path in sys.path:
            sys.path.remove(ast_path)
        
        modules_to_remove = [key for key in sys.modules.keys() if key == 'model' or key.startswith('model.')]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        if lcn_path not in sys.path:
            sys.path.insert(0, lcn_path)
        
        from model.lightcardiacnet import create_model as create_lcn_model
        
        self.model = create_lcn_model(
            model_type=config.get('model_type', 'ensemble'),
            input_size=config.get('input_size', 39),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print("LightCardiacNet model loaded successfully")
    
    def preprocess(self, audio_path: str) -> torch.Tensor:
        audio = load_audio(audio_path, target_sr=LCN_SAMPLE_RATE)
        audio = apply_bandpass_filter(audio, sr=LCN_SAMPLE_RATE)
        audio = normalize_audio(audio)
        
        target_length = int(LCN_TARGET_DURATION * LCN_SAMPLE_RATE)
        audio = pad_or_truncate(audio, target_length)
        
        features = extract_lcn_features(audio, sr=LCN_SAMPLE_RATE)
        
        return features.unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, audio_path: str) -> Dict:
        features = self.preprocess(audio_path)
        features = features.to(self.device)
        
        logits, _ = self.model(features)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        heart_sound_present = bool(probs[0] > 0.5)
        murmur_present = bool(probs[1] > 0.5) if heart_sound_present else False
        
        return {
            "predicted": 1 if murmur_present else 0,
            "murmur_prob": float(probs[1]),
            "heart_sound_prob": float(probs[0])
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
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
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def run_comparison(
    ast_model: ASTModelWrapper,
    lcn_model: LightCardiacNetWrapper,
    test_csv: str,
    test_dir: str,
    num_samples: int = 50,
    seed: int = 42
) -> Dict:
    """Run comparison between both models on same samples."""
    
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
    
    print(f"\nFound {len(valid_samples)} valid samples in test set")
    
    if num_samples > len(valid_samples):
        print(f"Requested {num_samples} samples but only {len(valid_samples)} available. Using all.")
        selected_samples = valid_samples
    else:
        selected_samples = random.sample(valid_samples, num_samples)
    
    print(f"Evaluating on {len(selected_samples)} samples...")
    print("-" * 80)
    
    y_true = []
    ast_preds = []
    lcn_preds = []
    ast_probs = []
    lcn_probs = []
    
    for i, sample in enumerate(selected_samples):
        try:
            ast_result = ast_model.predict(sample['file_path'])
            lcn_result = lcn_model.predict(sample['file_path'])
            
            ground_truth = sample['ground_truth']
            
            y_true.append(ground_truth)
            ast_preds.append(ast_result['predicted'])
            lcn_preds.append(lcn_result['predicted'])
            ast_probs.append(ast_result['murmur_prob'])
            lcn_probs.append(lcn_result['murmur_prob'])
            
            ast_correct = "✓" if ast_result['predicted'] == ground_truth else "✗"
            lcn_correct = "✓" if lcn_result['predicted'] == ground_truth else "✗"
            
            gt_label = "Murmur" if ground_truth == 1 else "Normal"
            ast_label = "Murmur" if ast_result['predicted'] == 1 else "Normal"
            lcn_label = "Murmur" if lcn_result['predicted'] == 1 else "Normal"
            
            print(f"[{i+1:3d}/{len(selected_samples)}] {sample['patient_id']}: GT={gt_label:7s} | "
                  f"AST={ast_label:7s}({ast_result['murmur_prob']:.2f}){ast_correct} | "
                  f"LCN={lcn_label:7s}({lcn_result['murmur_prob']:.2f}){lcn_correct}")
                  
        except Exception as e:
            print(f"[{i+1:3d}/{len(selected_samples)}] ERROR {sample['patient_id']}: {str(e)}")
    
    y_true = np.array(y_true)
    ast_preds = np.array(ast_preds)
    lcn_preds = np.array(lcn_preds)
    
    ast_metrics = compute_metrics(y_true, ast_preds)
    lcn_metrics = compute_metrics(y_true, lcn_preds)
    
    return {
        'ast_metrics': ast_metrics,
        'lcn_metrics': lcn_metrics,
        'total_samples': len(y_true),
        'positive_samples': int(np.sum(y_true == 1)),
        'negative_samples': int(np.sum(y_true == 0))
    }


def print_comparison_report(results: Dict):
    """Print formatted comparison report."""
    
    ast = results['ast_metrics']
    lcn = results['lcn_metrics']
    
    print("\n" + "=" * 80)
    print("               AST vs LightCardiacNet - MODEL COMPARISON REPORT")
    print("=" * 80)
    print(f"  Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    print("\n  DATASET SUMMARY")
    print("-" * 80)
    print(f"  Total Samples Evaluated: {results['total_samples']}")
    print(f"  Positive Samples (Murmur): {results['positive_samples']} ({100*results['positive_samples']/results['total_samples']:.1f}%)")
    print(f"  Negative Samples (Normal): {results['negative_samples']} ({100*results['negative_samples']/results['total_samples']:.1f}%)")
    
    print("\n  PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"  {'Metric':<30} {'AST':>15} {'LightCardiacNet':>17} {'Winner':>12}")
    print("  " + "-" * 76)
    
    metrics_to_compare = [
        ('Accuracy', 'accuracy', '%'),
        ('Balanced Accuracy', 'balanced_accuracy', '%'),
        ('Sensitivity (Recall)', 'sensitivity', '%'),
        ('Specificity', 'specificity', '%'),
        ('Precision', 'precision', '%'),
        ('NPV', 'npv', '%'),
        ('F1 Score', 'f1_score', '%'),
        ('MCC', 'mcc', '')
    ]
    
    ast_wins = 0
    lcn_wins = 0
    
    for name, key, unit in metrics_to_compare:
        ast_val = ast[key]
        lcn_val = lcn[key]
        
        if unit == '%':
            ast_str = f"{ast_val*100:.2f}%"
            lcn_str = f"{lcn_val*100:.2f}%"
        else:
            ast_str = f"{ast_val:.4f}"
            lcn_str = f"{lcn_val:.4f}"
        
        if ast_val > lcn_val + 0.001:
            winner = "AST ▲"
            ast_wins += 1
        elif lcn_val > ast_val + 0.001:
            winner = "LCN ▲"
            lcn_wins += 1
        else:
            winner = "TIE"
        
        print(f"  {name:<30} {ast_str:>15} {lcn_str:>17} {winner:>12}")
    
    print("\n  CONFUSION MATRICES")
    print("-" * 80)
    print("  AST Model:                               LightCardiacNet:")
    print("                    Predicted                             Predicted")
    print("                  Normal  Murmur                        Normal  Murmur")
    print(f"  Actual Normal   {ast['tn']:>6}  {ast['fp']:>6}         Actual Normal   {lcn['tn']:>6}  {lcn['fp']:>6}")
    print(f"         Murmur   {ast['fn']:>6}  {ast['tp']:>6}                Murmur   {lcn['fn']:>6}  {lcn['tp']:>6}")
    
    print("\n  VERDICT")
    print("-" * 80)
    
    if ast_wins > lcn_wins:
        verdict = "AST"
        margin = ast_wins - lcn_wins
    elif lcn_wins > ast_wins:
        verdict = "LightCardiacNet"
        margin = lcn_wins - ast_wins
    else:
        verdict = "TIE"
        margin = 0
    
    print(f"  Metrics won by AST: {ast_wins}/8")
    print(f"  Metrics won by LightCardiacNet: {lcn_wins}/8")
    
    if verdict != "TIE":
        print(f"\n  >>> WINNER: {verdict} (wins by {margin} metrics)")
    else:
        print(f"\n  >>> RESULT: TIE - Both models perform equally")
    
    key_diff_acc = (ast['accuracy'] - lcn['accuracy']) * 100
    key_diff_sens = (ast['sensitivity'] - lcn['sensitivity']) * 100
    key_diff_f1 = (ast['f1_score'] - lcn['f1_score']) * 100
    
    print(f"\n  Key Differences (AST - LCN):")
    print(f"    Accuracy:    {key_diff_acc:+.2f}%")
    print(f"    Sensitivity: {key_diff_sens:+.2f}%")
    print(f"    F1 Score:    {key_diff_f1:+.2f}%")
    
    print("\n  RECOMMENDATIONS")
    print("-" * 80)
    
    if ast['sensitivity'] >= lcn['sensitivity'] and ast['specificity'] >= lcn['specificity']:
        print("  AST model is superior in both sensitivity and specificity.")
        print("  Recommendation: Use AST model for production.")
    elif lcn['sensitivity'] >= ast['sensitivity'] and lcn['specificity'] >= ast['specificity']:
        print("  LightCardiacNet is superior in both sensitivity and specificity.")
        print("  Recommendation: Use LightCardiacNet for production.")
    elif ast['sensitivity'] > lcn['sensitivity']:
        print("  AST has better sensitivity (important for medical diagnosis).")
        print("  LightCardiacNet has better specificity (fewer false positives).")
        print("  Recommendation: Prioritize AST if missing murmurs is more costly.")
    else:
        print("  LightCardiacNet has better sensitivity (important for medical diagnosis).")
        print("  AST has better specificity (fewer false positives).")
        print("  Recommendation: Prioritize LightCardiacNet if missing murmurs is more costly.")
    
    print("\n" + "=" * 80)
    print("                              END OF REPORT")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare AST vs LightCardiacNet Models')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cpu')
    parser.add_argument('--all', action='store_true', help='Evaluate on all test samples')
    args = parser.parse_args()
    
    ast_model_path = os.path.join(SCRIPT_DIR, 'models', 'ASTmodel.pt')
    lcn_model_path = os.path.join(SCRIPT_DIR, 'models', 'lightcardiacnet.pt')
    
    data_dir = os.path.join(PROJECT_ROOT, 'DATA')
    test_csv = os.path.join(data_dir, 'cleaned_data_entries', 'test.csv')
    test_dir = os.path.join(data_dir, 'cleaned_data', 'test')
    
    for path, name in [(ast_model_path, "AST model"), (lcn_model_path, "LightCardiacNet model"),
                       (test_csv, "Test CSV"), (test_dir, "Test directory")]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at: {path}")
            sys.exit(1)
    
    print("=" * 80)
    print("           AST vs LightCardiacNet - MODEL COMPARISON")
    print("=" * 80)
    
    ast_model = ASTModelWrapper(ast_model_path, args.device)
    lcn_model = LightCardiacNetWrapper(lcn_model_path, args.device)
    
    num_samples = 999999 if args.all else args.samples
    
    results = run_comparison(
        ast_model=ast_model,
        lcn_model=lcn_model,
        test_csv=test_csv,
        test_dir=test_dir,
        num_samples=num_samples,
        seed=args.seed
    )
    
    print_comparison_report(results)
    
    return results


if __name__ == "__main__":
    main()
