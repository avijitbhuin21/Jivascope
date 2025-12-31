"""
Script to integrate PhysioNet 2016 heart sound dataset into existing cleaned_data structure.
"""
import os
import shutil
import csv
import random

random.seed(42)

BASE_DIR = r"d:\Programming\Projects\tunir_daa\Jivascope\NEW"
PHYSIONET_DIR = os.path.join(
    BASE_DIR,
    "classification-of-heart-sound-recordings-the-physionetcomputing-in-cardiology-challenge-2016-1.0.0",
    "classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
)

def extended_path(path):
    """Add Windows extended path prefix to handle paths > 260 chars."""
    if os.name == 'nt' and not path.startswith("\\\\?\\"):
        return "\\\\?\\" + os.path.abspath(path)
    return path

PHYSIONET_DIR = extended_path(PHYSIONET_DIR)

CLEANED_DATA_DIR = os.path.join(BASE_DIR, "cleaned_data")
CLEANED_ENTRIES_DIR = os.path.join(BASE_DIR, "cleaned_data_entries")

TRAIN_DIR = os.path.join(CLEANED_DATA_DIR, "train")
TEST_DIR = os.path.join(CLEANED_DATA_DIR, "test")
VAL_DIR = os.path.join(CLEANED_DATA_DIR, "val")

TRAIN_CSV = os.path.join(CLEANED_ENTRIES_DIR, "train.csv")
TEST_CSV = os.path.join(CLEANED_ENTRIES_DIR, "test.csv")
VAL_CSV = os.path.join(CLEANED_ENTRIES_DIR, "val.csv")

TRAINING_FOLDERS = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
VALIDATION_FOLDER = "validation"

TRAIN_SPLIT = 0.8


def parse_reference_csv(folder_path):
    reference_file = os.path.join(folder_path, "REFERENCE.csv")
    entries = []
    
    if not os.path.exists(reference_file):
        print(f"Warning: REFERENCE.csv not found in {folder_path}")
        return entries
    
    with open(reference_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                filename = row[0].strip()
                label = int(row[1].strip())
                outcome_label = 1 if label == 1 else 0
                entries.append((filename, outcome_label))
    
    return entries


def get_csv_header():
    return [
        "Patient ID", "Recording locations:", "Age", "Sex", "Height", "Weight",
        "Pregnancy status", "Murmur", "Murmur locations", "Most audible location",
        "Systolic murmur timing", "Systolic murmur shape", "Systolic murmur grading",
        "Systolic murmur pitch", "Systolic murmur quality", "Diastolic murmur timing",
        "Diastolic murmur shape", "Diastolic murmur grading", "Diastolic murmur pitch",
        "Diastolic murmur quality", "Outcome", "Campaign", "Additional ID",
        "Murmur_Label", "Outcome_Label"
    ]


def create_csv_row(patient_id, outcome_label):
    outcome_str = "Abnormal" if outcome_label == 1 else "Normal"
    row = [patient_id] + [""] * 19 + [outcome_str, "PhysioNet2016", ""] + ["", str(outcome_label)]
    return row


def copy_wav_file(src_folder, filename, dest_folder, prefix):
    src_path = os.path.join(src_folder, f"{filename}.wav")
    new_filename = f"physionet_{prefix}_{filename}.wav"
    dest_path = os.path.join(dest_folder, new_filename)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        return new_filename.replace(".wav", "")
    else:
        print(f"Warning: {src_path} not found")
        return None


def append_to_csv(csv_path, rows):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def process_training_folders():
    all_entries = []
    
    for folder_name in TRAINING_FOLDERS:
        folder_path = os.path.join(PHYSIONET_DIR, folder_name)
        prefix = folder_name.split("-")[1]
        
        entries = parse_reference_csv(folder_path)
        for filename, outcome_label in entries:
            all_entries.append((folder_path, filename, outcome_label, prefix))
    
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * TRAIN_SPLIT)
    train_entries = all_entries[:split_idx]
    test_entries = all_entries[split_idx:]
    
    return train_entries, test_entries


def process_validation_folder():
    folder_path = os.path.join(PHYSIONET_DIR, VALIDATION_FOLDER)
    entries = parse_reference_csv(folder_path)
    return [(folder_path, filename, outcome_label, "val") for filename, outcome_label in entries]


def main():
    print("=" * 60)
    print("PhysioNet 2016 Data Integration Script")
    print("=" * 60)
    
    train_entries, test_entries = process_training_folders()
    val_entries = process_validation_folder()
    
    print(f"\nProcessing {len(train_entries)} files for train...")
    train_rows = []
    train_copied = 0
    for folder_path, filename, outcome_label, prefix in train_entries:
        patient_id = copy_wav_file(folder_path, filename, TRAIN_DIR, prefix)
        if patient_id:
            train_rows.append(create_csv_row(patient_id, outcome_label))
            train_copied += 1
    
    print(f"Processing {len(test_entries)} files for test...")
    test_rows = []
    test_copied = 0
    for folder_path, filename, outcome_label, prefix in test_entries:
        patient_id = copy_wav_file(folder_path, filename, TEST_DIR, prefix)
        if patient_id:
            test_rows.append(create_csv_row(patient_id, outcome_label))
            test_copied += 1
    
    print(f"Processing {len(val_entries)} files for val...")
    val_rows = []
    val_copied = 0
    for folder_path, filename, outcome_label, prefix in val_entries:
        patient_id = copy_wav_file(folder_path, filename, VAL_DIR, prefix)
        if patient_id:
            val_rows.append(create_csv_row(patient_id, outcome_label))
            val_copied += 1
    
    print("\nUpdating CSV files...")
    append_to_csv(TRAIN_CSV, train_rows)
    append_to_csv(TEST_CSV, test_rows)
    append_to_csv(VAL_CSV, val_rows)
    
    print("\n" + "=" * 60)
    print("=== PhysioNet 2016 Data Integration Complete ===")
    print("=" * 60)
    print(f"Training files copied to train/: {train_copied}")
    print(f"Training files copied to test/: {test_copied}")
    print(f"Validation files copied to val/: {val_copied}")
    print(f"Train CSV entries added: {len(train_rows)}")
    print(f"Test CSV entries added: {len(test_rows)}")
    print(f"Val CSV entries added: {len(val_rows)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
