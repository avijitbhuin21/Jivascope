"""
Data Exploration Script for CirCor Heart Sound Dataset
Analyzes the cleaned dataset and generates summary statistics
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not installed. Audio analysis will be skipped.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed. Plots will be skipped.")


BASE_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = BASE_DIR / 'the-circor-digiscope-phonocardiogram-dataset-1.0.3'
CLEANED_DIR = BASE_DIR / 'cleaned_data'
TRAINING_DATA_DIR = DATASET_DIR / 'training_data'


def load_cleaned_data():
    train_df = pd.read_csv(CLEANED_DIR / 'train.csv')
    val_df = pd.read_csv(CLEANED_DIR / 'val.csv')
    test_df = pd.read_csv(CLEANED_DIR / 'test.csv')
    full_df = pd.read_csv(CLEANED_DIR / 'cleaned_data.csv')
    return train_df, val_df, test_df, full_df


def print_dataset_statistics(train_df, val_df, test_df, full_df):
    print("=" * 70)
    print("JIVASCOPE - DATA EXPLORATION REPORT")
    print("=" * 70)
    
    print("\n📊 DATASET SPLIT SUMMARY")
    print("-" * 40)
    print(f"{'Split':<15} {'Patients':<12} {'Percentage':<12}")
    print("-" * 40)
    print(f"{'Training':<15} {len(train_df):<12} {len(train_df)/len(full_df)*100:.1f}%")
    print(f"{'Validation':<15} {len(val_df):<12} {len(val_df)/len(full_df)*100:.1f}%")
    print(f"{'Test':<15} {len(test_df):<12} {len(test_df)/len(full_df)*100:.1f}%")
    print("-" * 40)
    print(f"{'TOTAL':<15} {len(full_df):<12} 100.0%")


def print_class_distribution(full_df, train_df, val_df, test_df):
    print("\n\n📈 MURMUR CLASS DISTRIBUTION")
    print("-" * 60)
    print(f"{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    for label_val, label_name in [(0, 'Absent'), (1, 'Present')]:
        train_count = (train_df['Murmur_Label'] == label_val).sum()
        val_count = (val_df['Murmur_Label'] == label_val).sum()
        test_count = (test_df['Murmur_Label'] == label_val).sum()
        total_count = (full_df['Murmur_Label'] == label_val).sum()
        print(f"{label_name:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}")
    
    print("\n\n📈 OUTCOME CLASS DISTRIBUTION")
    print("-" * 60)
    print(f"{'Class':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    for label_val, label_name in [(0, 'Normal'), (1, 'Abnormal')]:
        train_count = (train_df['Outcome_Label'] == label_val).sum()
        val_count = (val_df['Outcome_Label'] == label_val).sum()
        test_count = (test_df['Outcome_Label'] == label_val).sum()
        total_count = (full_df['Outcome_Label'] == label_val).sum()
        print(f"{label_name:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}")


def calculate_class_weights(train_df):
    print("\n\n⚖️ CLASS IMBALANCE ANALYSIS")
    print("-" * 50)
    
    murmur_counts = train_df['Murmur_Label'].value_counts()
    murmur_ratio = murmur_counts[0] / murmur_counts[1]
    print(f"Murmur Absent:Present ratio = {murmur_ratio:.2f}:1")
    
    murmur_weights = len(train_df) / (2 * murmur_counts)
    print(f"Suggested Murmur weights: Absent={murmur_weights[0]:.3f}, Present={murmur_weights[1]:.3f}")
    
    outcome_counts = train_df['Outcome_Label'].value_counts()
    outcome_ratio = outcome_counts[0] / outcome_counts[1] if 0 in outcome_counts and 1 in outcome_counts else 1.0
    print(f"\nOutcome Normal:Abnormal ratio = {outcome_ratio:.2f}:1")
    
    outcome_weights = len(train_df) / (2 * outcome_counts)
    print(f"Suggested Outcome weights: Normal={outcome_weights[0]:.3f}, Abnormal={outcome_weights[1]:.3f}")
    
    return {
        'murmur_weights': murmur_weights.to_dict(),
        'outcome_weights': outcome_weights.to_dict()
    }


def analyze_audio_files(full_df, sample_size=50):
    if not LIBROSA_AVAILABLE:
        print("\n\n⚠️ AUDIO ANALYSIS SKIPPED (librosa not installed)")
        return None
    
    print("\n\n🎵 AUDIO FILE ANALYSIS")
    print("-" * 50)
    
    durations = []
    sample_rates = []
    errors = []
    
    patient_ids = full_df['Patient ID'].sample(min(sample_size, len(full_df)), random_state=42).tolist()
    
    print(f"Analyzing {len(patient_ids)} sample audio files...")
    
    for pid in patient_ids:
        wav_files = list(TRAINING_DATA_DIR.glob(f"{pid}*.wav"))
        for wav_file in wav_files[:1]:
            try:
                y, sr = librosa.load(wav_file, sr=None)
                duration = len(y) / sr
                durations.append(duration)
                sample_rates.append(sr)
            except Exception as e:
                errors.append((wav_file.name, str(e)))
    
    if durations:
        print(f"\nSample Rate: {sample_rates[0]} Hz (expected: 4000 Hz)")
        print(f"Duration Statistics (from {len(durations)} files):")
        print(f"  Min: {min(durations):.2f}s")
        print(f"  Max: {max(durations):.2f}s")
        print(f"  Mean: {np.mean(durations):.2f}s")
        print(f"  Median: {np.median(durations):.2f}s")
    
    if errors:
        print(f"\n⚠️ Errors loading {len(errors)} files")
    
    return {'durations': durations, 'sample_rates': sample_rates, 'errors': errors}


def analyze_tsv_segmentation(full_df, sample_size=10):
    print("\n\n🔍 TSV SEGMENTATION FILE ANALYSIS")
    print("-" * 50)
    
    patient_ids = full_df['Patient ID'].sample(min(sample_size, len(full_df)), random_state=42).tolist()
    
    segment_stats = {'S1': 0, 'S2': 0, 'systole': 0, 'diastole': 0}
    total_segments = 0
    
    for pid in patient_ids:
        tsv_files = list(TRAINING_DATA_DIR.glob(f"{pid}*.tsv"))
        for tsv_file in tsv_files[:1]:
            try:
                tsv_df = pd.read_csv(tsv_file, sep='\t', header=None, names=['start', 'end', 'state'])
                for state in segment_stats.keys():
                    segment_stats[state] += (tsv_df['state'] == state).sum()
                total_segments += len(tsv_df)
            except Exception:
                pass
    
    if total_segments > 0:
        print(f"Segment distribution (from {sample_size} patients):")
        for state, count in segment_stats.items():
            pct = count / total_segments * 100
            print(f"  {state}: {count} ({pct:.1f}%)")


def create_visualizations(full_df, train_df, output_dir=None):
    if not PLOTS_AVAILABLE:
        print("\n\n⚠️ VISUALIZATIONS SKIPPED (matplotlib not installed)")
        return
    
    if output_dir is None:
        output_dir = BASE_DIR / 'notebooks'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n\n📊 GENERATING VISUALIZATIONS")
    print("-" * 50)
    
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    murmur_counts = full_df['Murmur'].value_counts()
    axes[0].bar(murmur_counts.index, murmur_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Murmur Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Murmur Status')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(murmur_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    outcome_counts = full_df['Outcome'].value_counts()
    axes[1].bar(outcome_counts.index, outcome_counts.values, color=['#3498db', '#e67e22'])
    axes[1].set_title('Outcome Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Outcome')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(outcome_counts.values):
        axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / 'class_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


def main():
    print("\nLoading cleaned datasets...")
    train_df, val_df, test_df, full_df = load_cleaned_data()
    
    print_dataset_statistics(train_df, val_df, test_df, full_df)
    print_class_distribution(full_df, train_df, val_df, test_df)
    class_weights = calculate_class_weights(train_df)
    analyze_audio_files(full_df, sample_size=30)
    analyze_tsv_segmentation(full_df, sample_size=10)
    create_visualizations(full_df, train_df)
    
    print("\n" + "=" * 70)
    print("DATA EXPLORATION COMPLETE!")
    print("=" * 70)
    
    return {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'class_weights': class_weights
    }


if __name__ == "__main__":
    main()
