# 🫀 Jivascope Training Guide

This guide explains how to train the heart sound classification model after cloning the repository.

---

## Quick Start (Local Machine with GPU)

```bash
# 1. Clone the repository
git clone <your-repo-url> Jivascope
cd Jivascope

# 2. Create virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset (if not included in repo)
# Place the CirCor dataset in: the-circor-digiscope-phonocardiogram-dataset-1.0.3/

# 5. Run training
python train.py
```

---

## Training on Google Colab (Recommended)

### Step 1: Set up Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)

### Step 2: Clone Repo and Install Dependencies

```python
# Clone your repository
!git clone <your-repo-url> Jivascope
%cd Jivascope

# Install dependencies (Colab already has most, but some are missing)
!pip install audiomentations wfdb pywavelets -q
```

### Step 3: Upload or Download Dataset

**Option A: Mount Google Drive (if dataset is on Drive)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset from Drive
!cp -r "/content/drive/MyDrive/datasets/the-circor-digiscope-phonocardiogram-dataset-1.0.3" .
```

**Option B: Download from PhysioNet**
```python
!wget -q https://physionet.org/static/published-projects/circor-heart-sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip
!unzip -q the-circor-digiscope-phonocardiogram-dataset-1.0.3.zip
```

### Step 4: Run Training

```python
# Train with default settings (50 epochs)
!python train.py

# Or with custom settings
!python train.py --epochs 30 --batch_size 32
```

### Step 5: Monitor Training with TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir logs
```

### Step 6: Download Trained Model

```python
from google.colab import files
files.download('checkpoints/best_model.pt')
```

---

## Training Script Options

```bash
python train.py [OPTIONS]

Options:
  --epochs N          Number of training epochs (default: 50)
  --batch_size N      Batch size (default: 16)
  --lr FLOAT          Learning rate (default: 0.0001)
  --device DEVICE     cuda or cpu (default: cuda if available)
  --backbone NAME     efficientnet_b0, resnet18, or resnet34
  --no_pretrained     Don't use pretrained ImageNet weights
  --resume PATH       Resume from a checkpoint file
```

### Examples

```bash
# Quick test run (5 epochs, CPU)
python train.py --epochs 5 --device cpu

# Full training with larger batch (needs more GPU memory)
python train.py --epochs 100 --batch_size 32

# Try different backbone
python train.py --backbone resnet18

# Resume interrupted training
python train.py --resume checkpoints/best_model.pt
```

---

## Expected Output

```
============================================================
🫀 JIVASCOPE - Heart Sound Classification
============================================================
📂 Loading data...
  Train samples: 611 patients
  Val samples: 131 patients
  Test samples: 132 patients

📊 Computing class weights...
  Murmur weights: [0.89, 3.45]
  Outcome weights: [0.95, 1.06]

🔄 Creating data loaders...
  Train batches: 152
  Val batches: 33

🤖 Creating model...
  Backbone: efficientnet_b0
  Total parameters: 5,288,548
  Trainable parameters: 5,288,548

============================================================
🚀 Starting training...
============================================================

Epoch 1/50 (32.5s)
  Train - Loss: 1.2345 | Murmur Acc: 0.6823 | Outcome Acc: 0.5912
  Val   - Loss: 0.9876 | Murmur Acc: 0.7234 | Outcome Acc: 0.6123
  💾 New best model saved! (val_loss: 0.9876)

...

============================================================
✅ Training complete!
============================================================

📁 Checkpoints saved to: checkpoints/
📊 TensorBoard logs: logs/
```

---

## Output Files

After training, you'll find:

| File | Description |
|------|-------------|
| `checkpoints/best_model.pt` | Best model (lowest validation loss) |
| `checkpoints/final_model.pt` | Model from last epoch |
| `logs/` | TensorBoard training logs |

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train.py --batch_size 8
```

### CUDA Not Available
```bash
# Force CPU training (slow but works)
python train.py --device cpu
```

### Missing Dataset
Ensure the dataset structure is:
```
Jivascope/
├── the-circor-digiscope-phonocardiogram-dataset-1.0.3/
│   └── training_data/
│       ├── 2530_AV.wav
│       ├── 2530_MV.wav
│       └── ...
└── cleaned_data/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## Full Colab Notebook Template

```python
# ============================================
# 🫀 Jivascope Training - Google Colab
# ============================================

# 1. Check GPU
!nvidia-smi

# 2. Clone repository
!git clone <your-repo-url> Jivascope
%cd Jivascope

# 3. Install missing packages
!pip install audiomentations wfdb pywavelets -q

# 4. Mount Drive and copy dataset
from google.colab import drive
drive.mount('/content/drive')
!cp -r "/content/drive/MyDrive/path/to/dataset" .

# 5. Verify data
!ls cleaned_data/
!ls the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/ | head -20

# 6. Run training
!python train.py --epochs 50

# 7. Monitor with TensorBoard
%load_ext tensorboard
%tensorboard --logdir logs

# 8. Copy model to Drive
!cp checkpoints/best_model.pt "/content/drive/MyDrive/jivascope_models/"

# 9. Download model locally
from google.colab import files
files.download('checkpoints/best_model.pt')
```

---

## Next Steps After Training

1. **Evaluate on Test Set**: Run inference on `test.csv` (Step 6)
2. **Export Model**: Convert to ONNX for deployment
3. **API Integration**: Create FastAPI endpoint (Step 7)
