# Step 1: Environment Setup & Data Exploration

## Objective
Set up the development environment, install dependencies, and perform exploratory data analysis (EDA) on the CirCor dataset.

## Prerequisites
- Python 3.10+ installed
- Google Colab Pro account (for GPU training)
- CirCor dataset downloaded to `the-circor-digiscope-phonocardiogram-dataset-1.0.3/`

---

## Implementation Details

### 1.1 Create Virtual Environment (Local)

```bash
cd Jivascope
python -m venv venv
venv\Scripts\activate   # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.2 Create requirements.txt

Create the requirements file with all dependencies as specified in `01_tech_stack.md`.

### 1.3 Data Exploration Script

Create `notebooks/01_data_exploration.ipynb` or `src/data/explore.py` to:

1. **Load and parse training_data.csv**
   - Count total patients
   - Analyze class distribution (Murmur: Present/Absent/Unknown)
   - Analyze outcome distribution (Normal/Abnormal)

2. **Analyze audio files**
   - Sample rate verification (should be 4000 Hz)
   - Duration distribution
   - Identify corrupted or missing files

3. **Visualizations**
   - Murmur class distribution bar chart
   - Outcome class distribution bar chart
   - Audio duration histogram
   - Sample waveform plots (normal vs abnormal)
   - Sample spectrogram plots

4. **Handle "Unknown" murmur cases**
   - Decide strategy: exclude from training or treat as separate class

### 1.4 Parse TSV Segmentation Files

Understand the TSV structure:
- Columns: `start_time`, `end_time`, `state` (S1, systole, S2, diastole)
- This will be used for training-time data augmentation

---

## Research Areas
- Explore `wfdb` library for reading `.hea` header files
- Understand WFDB format used in PhysioNet datasets
- Research data augmentation strategies for audio classification

---

## Expected Outcome
- Working Python environment with all dependencies
- Comprehensive understanding of dataset characteristics
- Identification of class imbalance issues
- Decision on handling "Unknown" murmur cases

---

## Estimated Effort
- **2-3 hours** for environment setup
- **3-4 hours** for complete EDA

---

## Dependencies
- None (first step)
