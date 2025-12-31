"""
LightCardiacNet Evaluation Script

Evaluates trained model on test set with detailed metrics.

Usage:
    python evaluate.py                   # Use default paths
    python evaluate.py --model path/to/model.pt --test-csv path/to/test.csv
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.lightcardiacnet import create_model
from model.dataset import HeartSoundDataset
from torch.utils.data import DataLoader


def evaluate_model(model_path: str, test_csv: str, test_dir: str, device: str = 'cpu'):
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = create_model(
        model_type=config.get('model_type', 'ensemble'),
        input_size=config.get('input_size', 39),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Loading test dataset...")
    test_dataset = HeartSoundDataset(test_csv, test_dir, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_dataset)}")
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating"):
            features = features.to(device)
            logits, _ = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n--- HEART SOUND DETECTION ---")
    print(f"Accuracy:  {accuracy_score(all_labels[:, 0], all_preds[:, 0]):.4f}")
    print(f"Precision: {precision_score(all_labels[:, 0], all_preds[:, 0], zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels[:, 0], all_preds[:, 0], zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(all_labels[:, 0], all_preds[:, 0], zero_division=0):.4f}")
    
    print("\n--- MURMUR DETECTION ---")
    print(f"Accuracy:  {accuracy_score(all_labels[:, 1], all_preds[:, 1]):.4f}")
    print(f"Precision: {precision_score(all_labels[:, 1], all_preds[:, 1], zero_division=0):.4f}")
    print(f"Recall:    {recall_score(all_labels[:, 1], all_preds[:, 1], zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(all_labels[:, 1], all_preds[:, 1], zero_division=0):.4f}")
    
    try:
        auc = roc_auc_score(all_labels[:, 1], all_probs[:, 1])
        print(f"AUC-ROC:   {auc:.4f}")
    except ValueError:
        print("AUC-ROC:   N/A (insufficient classes)")
    
    print("\n--- CONFUSION MATRIX (Murmur) ---")
    cm = confusion_matrix(all_labels[:, 1], all_preds[:, 1])
    print(f"              Pred Absent  Pred Present")
    print(f"True Absent       {cm[0, 0]:5d}         {cm[0, 1]:5d}")
    print(f"True Present      {cm[1, 0]:5d}         {cm[1, 1]:5d}")
    
    print("\n--- OVERALL METRICS ---")
    overall_acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
    overall_f1 = f1_score(all_labels.flatten(), all_preds.flatten(), average='macro')
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Macro F1 Score:   {overall_f1:.4f}")
    
    print("="*60)
    
    return {
        'heart_accuracy': accuracy_score(all_labels[:, 0], all_preds[:, 0]),
        'murmur_accuracy': accuracy_score(all_labels[:, 1], all_preds[:, 1]),
        'murmur_f1': f1_score(all_labels[:, 1], all_preds[:, 1], zero_division=0),
        'murmur_recall': recall_score(all_labels[:, 1], all_preds[:, 1], zero_division=0),
        'overall_accuracy': overall_acc
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate LightCardiacNet')
    parser.add_argument('--model', type=str, default='model/checkpoints/best_acc_model.pt')
    parser.add_argument('--test-csv', type=str, default='cleaned_data_entries/test.csv')
    parser.add_argument('--test-dir', type=str, default='cleaned_data/test')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    evaluate_model(args.model, args.test_csv, args.test_dir, args.device)


if __name__ == "__main__":
    main()
