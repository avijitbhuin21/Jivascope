# Step 1: Environment Setup & Data Verification

## Objective
Set up the development environment and verify data integrity.

## Tasks

### 1.1 Create Virtual Environment
```bash
cd d:\Programming\Projects\tunir_daa\Jivascope\NEW
python -m venv venv
venv\Scripts\activate
pip install -r planning/01_requirements.txt
```

### 1.2 Verify Data Structure
- [ ] Confirm all audio files exist in `cleaned_data/train`, `test`, `val`
- [ ] Verify CSV files match audio files
- [ ] Check for missing or corrupted files

### 1.3 Data Statistics
Run initial analysis:
- Total patients: ~875
- Train/Val/Test split verification
- Class distribution (Normal vs Murmur)
- Audio duration statistics

### 1.4 Create Model Directory Structure
```
NEW/model/
├── __init__.py
├── lightcardiacnet.py
├── attention.py
├── dataset.py
├── trainer.py
└── utils.py
```

## Deliverables
- [ ] Working virtual environment
- [ ] Data integrity report
- [ ] Directory structure created

## Estimated Time
~1-2 hours
