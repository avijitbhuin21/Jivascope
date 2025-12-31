"""
Data Cleaning and Train/Test Split Script
Removes 'Unknown' murmur samples and creates train/test datasets
"""

import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'the-circor-digiscope-phonocardiogram-dataset-1.0.3')
OUTPUT_DIR = os.path.join(BASE_DIR, 'cleaned_data')

def main():
    print("=" * 60)
    print("JIVASCOPE DATA CLEANING SCRIPT")
    print("=" * 60)
    
    csv_path = os.path.join(DATASET_DIR, 'training_data.csv')
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 ORIGINAL DATA STATISTICS")
    print(f"Total patients: {len(df)}")
    print(f"\nMurmur distribution:")
    print(df['Murmur'].value_counts())
    print(f"\nOutcome distribution:")
    print(df['Outcome'].value_counts())
    
    unknown_count = (df['Murmur'] == 'Unknown').sum()
    print(f"\n⚠️  Samples with 'Unknown' murmur: {unknown_count}")
    
    df_clean = df[df['Murmur'] != 'Unknown'].copy()
    df_clean['Murmur_Label'] = (df_clean['Murmur'] == 'Present').astype(int)
    df_clean['Outcome_Label'] = (df_clean['Outcome'] == 'Abnormal').astype(int)
    
    print(f"\n✅ CLEANED DATA STATISTICS")
    print(f"Total patients after cleaning: {len(df_clean)}")
    print(f"Removed patients: {len(df) - len(df_clean)}")
    print(f"\nMurmur distribution (cleaned):")
    print(df_clean['Murmur'].value_counts())
    print(f"\nOutcome distribution (cleaned):")
    print(df_clean['Outcome'].value_counts())
    
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.15,
        stratify=df_clean['Murmur_Label'],
        random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.176,  # 0.15 / 0.85 ≈ 0.176 to get 15% of original
        stratify=train_df['Murmur_Label'],
        random_state=42
    )
    
    print(f"\n📂 TRAIN/VAL/TEST SPLIT (by patient)")
    print(f"Training set: {len(train_df)} patients ({len(train_df)/len(df_clean)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} patients ({len(val_df)/len(df_clean)*100:.1f}%)")
    print(f"Test set: {len(test_df)} patients ({len(test_df)/len(df_clean)*100:.1f}%)")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_clean.to_csv(os.path.join(OUTPUT_DIR, 'cleaned_data.csv'), index=False)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    
    print(f"\n💾 FILES SAVED TO: {OUTPUT_DIR}")
    print(f"  - cleaned_data.csv (all cleaned data)")
    print(f"  - train.csv ({len(train_df)} patients)")
    print(f"  - val.csv ({len(val_df)} patients)")
    print(f"  - test.csv ({len(test_df)} patients)")
    
    print(f"\n📊 LABEL ENCODING")
    print(f"  Murmur: 0=Absent, 1=Present")
    print(f"  Outcome: 0=Normal, 1=Abnormal")
    
    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
