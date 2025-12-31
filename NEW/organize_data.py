"""
Script to organize the cleaned data into a structured folder format.
Creates folders for train, test, val with WAV files and a separate folder for CSV entries.
"""

import os
import shutil
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"d:\Programming\Projects\tunir_daa\Jivascope")
SOURCE_DATA_DIR = BASE_DIR / "the-circor-digiscope-phonocardiogram-dataset-1.0.3" / "training_data"
OLD_CLEANED_DATA = BASE_DIR / "OLD" / "cleaned_data"
NEW_DIR = BASE_DIR / "NEW"

CLEANED_DATA_DIR = NEW_DIR / "cleaned_data"
CLEANED_DATA_ENTRIES_DIR = NEW_DIR / "cleaned_data_entries"


def create_directory_structure():
    """Create the new folder structure."""
    folders = [
        CLEANED_DATA_DIR / "train",
        CLEANED_DATA_DIR / "test",
        CLEANED_DATA_DIR / "val",
        CLEANED_DATA_ENTRIES_DIR
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Created: {folder}")


def copy_csv_files():
    """Copy CSV files to cleaned_data_entries folder."""
    csv_files = ["train.csv", "test.csv", "val.csv", "cleaned_data.csv"]
    
    for csv_file in csv_files:
        src = OLD_CLEANED_DATA / csv_file
        dst = CLEANED_DATA_ENTRIES_DIR / csv_file
        
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied: {csv_file} -> {dst}")
        else:
            print(f"Warning: {csv_file} not found at {src}")


def get_wav_files_for_patient(patient_id):
    """Find all WAV files for a given patient ID."""
    wav_files = []
    patient_id_str = str(patient_id)
    
    for file in SOURCE_DATA_DIR.iterdir():
        if file.suffix.lower() == '.wav' and file.stem.startswith(f"{patient_id_str}_"):
            wav_files.append(file)
    
    return wav_files


def copy_wav_files_for_split(split_name):
    """Copy WAV files for a specific data split (train, test, val)."""
    csv_path = OLD_CLEANED_DATA / f"{split_name}.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    target_dir = CLEANED_DATA_DIR / split_name
    
    total_patients = len(df)
    copied_files = 0
    missing_patients = []
    
    print(f"\nProcessing {split_name} split ({total_patients} patients)...")
    
    for idx, row in df.iterrows():
        patient_id = row['Patient ID']
        wav_files = get_wav_files_for_patient(patient_id)
        
        if not wav_files:
            missing_patients.append(patient_id)
            continue
        
        for wav_file in wav_files:
            dst = target_dir / wav_file.name
            if not dst.exists():
                shutil.copy2(wav_file, dst)
                copied_files += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{total_patients} patients...")
    
    print(f"  Completed: {copied_files} WAV files copied to {split_name}")
    
    if missing_patients:
        print(f"  Warning: {len(missing_patients)} patients had no WAV files found")


def main():
    print("="*60)
    print("Organizing Jivascope Data")
    print("="*60)
    
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    print("\n2. Copying CSV files...")
    copy_csv_files()
    
    print("\n3. Copying WAV files for each split...")
    for split in ["train", "test", "val"]:
        copy_wav_files_for_split(split)
    
    print("\n" + "="*60)
    print("Data organization complete!")
    print("="*60)
    
    for split in ["train", "test", "val"]:
        folder = CLEANED_DATA_DIR / split
        wav_count = len(list(folder.glob("*.wav")))
        print(f"  {split}: {wav_count} WAV files")


if __name__ == "__main__":
    main()
