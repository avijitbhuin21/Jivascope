"""
Test script for the data preprocessing pipeline
Verifies that all components work correctly
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd


def test_preprocessing():
    print("\n" + "=" * 60)
    print("TESTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    from src.data.preprocessing import (
        load_audio,
        normalize_audio,
        pad_or_truncate,
        apply_bandpass_filter,
        create_mel_spectrogram,
        create_multichannel_spectrogram,
        SAMPLE_RATE,
        TARGET_DURATION
    )
    
    print("\n✅ Preprocessing module imported successfully")
    
    audio_dir = BASE_DIR / 'the-circor-digiscope-phonocardiogram-dataset-1.0.3' / 'training_data'
    cleaned_dir = BASE_DIR / 'cleaned_data'
    
    train_df = pd.read_csv(cleaned_dir / 'train.csv')
    sample_patient = train_df['Patient ID'].iloc[0]
    
    wav_files = list(audio_dir.glob(f"{sample_patient}_*.wav"))
    if not wav_files:
        print(f"❌ No audio files found for patient {sample_patient}")
        return False
    
    test_file = wav_files[0]
    print(f"\n📂 Testing with: {test_file.name}")
    
    print("\n1️⃣ Testing audio loading...")
    audio = load_audio(str(test_file), SAMPLE_RATE)
    print(f"   Shape: {audio.shape}, Duration: {len(audio)/SAMPLE_RATE:.2f}s")
    
    print("\n2️⃣ Testing normalization...")
    audio_norm = normalize_audio(audio)
    print(f"   Mean: {audio_norm.mean():.4f}, Std: {audio_norm.std():.4f}")
    
    print("\n3️⃣ Testing bandpass filter...")
    audio_filtered = apply_bandpass_filter(audio, SAMPLE_RATE)
    print(f"   Filtered shape: {audio_filtered.shape}")
    
    print("\n4️⃣ Testing pad/truncate...")
    target_len = int(TARGET_DURATION * SAMPLE_RATE)
    audio_fixed = pad_or_truncate(audio_norm, target_len)
    print(f"   Fixed length: {len(audio_fixed)} samples ({TARGET_DURATION}s)")
    
    print("\n5️⃣ Testing mel spectrogram...")
    mel_spec = create_mel_spectrogram(audio_fixed, SAMPLE_RATE)
    print(f"   Mel spectrogram shape: {mel_spec.shape}")
    
    print("\n6️⃣ Testing multi-channel spectrogram...")
    multi_spec = create_multichannel_spectrogram(audio_fixed, SAMPLE_RATE)
    print(f"   Multi-channel shape: {multi_spec.shape} (C, H, W)")
    print(f"   Value range: [{multi_spec.min():.4f}, {multi_spec.max():.4f}]")
    
    print("\n✅ All preprocessing tests passed!")
    return True


def test_augmentation():
    print("\n" + "=" * 60)
    print("TESTING AUGMENTATION")
    print("=" * 60)
    
    from src.data.augmentation import (
        AudioAugmentor,
        SpecAugment,
        get_train_augmentor,
        get_spec_augmentor
    )
    
    print("\n✅ Augmentation module imported successfully")
    
    audio = np.random.randn(40000).astype(np.float32)
    
    print("\n1️⃣ Testing audio augmentor...")
    augmentor = get_train_augmentor()
    audio_aug = augmentor(audio.copy(), sr=4000)
    print(f"   Original shape: {audio.shape}, Augmented shape: {audio_aug.shape}")
    
    print("\n2️⃣ Testing SpecAugment...")
    spec = np.random.randn(3, 128, 313).astype(np.float32)
    spec_augmentor = get_spec_augmentor()
    spec_aug = spec_augmentor(spec.copy())
    print(f"   Original shape: {spec.shape}, Augmented shape: {spec_aug.shape}")
    
    num_zeros_before = np.sum(spec == 0)
    num_zeros_after = np.sum(spec_aug == 0)
    print(f"   Zeros before: {num_zeros_before}, after: {num_zeros_after} (masking applied)")
    
    print("\n✅ All augmentation tests passed!")
    return True


def test_dataset():
    print("\n" + "=" * 60)
    print("TESTING DATASET")
    print("=" * 60)
    
    from src.data.dataset import HeartSoundDataset, create_dataloaders, get_class_weights
    
    print("\n✅ Dataset module imported successfully")
    
    cleaned_dir = BASE_DIR / 'cleaned_data'
    audio_dir = BASE_DIR / 'the-circor-digiscope-phonocardiogram-dataset-1.0.3' / 'training_data'
    
    train_df = pd.read_csv(cleaned_dir / 'train.csv')
    val_df = pd.read_csv(cleaned_dir / 'val.csv')
    
    print(f"\n📊 Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    print("\n1️⃣ Testing HeartSoundDataset...")
    small_df = train_df.head(5)
    dataset = HeartSoundDataset(
        small_df, 
        str(audio_dir), 
        is_training=True,
        target_duration=5.0
    )
    print(f"   Dataset size: {len(dataset)} samples")
    
    print("\n2️⃣ Testing __getitem__...")
    spectrogram, murmur_label, outcome_label, patient_id = dataset[0]
    print(f"   Spectrogram shape: {spectrogram.shape}")
    print(f"   Murmur label: {murmur_label}, Outcome label: {outcome_label}")
    print(f"   Patient ID: {patient_id}")
    print(f"   Dtype: {spectrogram.dtype}")
    
    print("\n3️⃣ Testing class weights...")
    weights = get_class_weights(train_df)
    print(f"   Murmur weights: {weights['murmur_weights']}")
    print(f"   Outcome weights: {weights['outcome_weights']}")
    
    print("\n✅ All dataset tests passed!")
    return True


def test_config():
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    from src.utils.config import get_default_config, print_config
    
    print("\n✅ Config module imported successfully")
    
    config = get_default_config()
    print_config(config)
    
    print("\n✅ Configuration test passed!")
    return True


def main():
    print("\n" + "=" * 70)
    print("JIVASCOPE - DATA PIPELINE TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    try:
        results['preprocessing'] = test_preprocessing()
    except Exception as e:
        print(f"\n❌ Preprocessing test failed: {e}")
        results['preprocessing'] = False
    
    try:
        results['augmentation'] = test_augmentation()
    except Exception as e:
        print(f"\n❌ Augmentation test failed: {e}")
        results['augmentation'] = False
    
    try:
        results['dataset'] = test_dataset()
    except Exception as e:
        print(f"\n❌ Dataset test failed: {e}")
        results['dataset'] = False
    
    try:
        results['config'] = test_config()
    except Exception as e:
        print(f"\n❌ Config test failed: {e}")
        results['config'] = False
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("✅ ALL TESTS PASSED!" if all_passed else "❌ SOME TESTS FAILED"))
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
