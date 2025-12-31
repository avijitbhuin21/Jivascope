# Requirements Document: Heart Sound Classification (Jivascope)

## Project Vision
Build a machine learning model to classify heart sounds from the CirCor DigiScope Phonocardiogram Dataset, detecting:
1. **Murmur Presence**: Absent / Present (binary) - *Unknown samples excluded*
2. **Clinical Outcome**: Normal / Abnormal (binary)

## Dataset Overview
- **Source**: CirCor DigiScope Phonocardiogram Dataset v1.0.3
- **Original Patients**: 942 patients
- **Cleaned Patients**: 874 patients (68 "Unknown" murmur samples removed)
- **Audio Files**: ~7,215 files (4 auscultation locations per patient: AV, PV, TV, MV)
- **Sample Rate**: 4000 Hz
- **Format**: WAV audio, HEA headers, TSV segmentation annotations
- **Split**: Train (611) / Val (131) / Test (132) patients

## User Requirements

| Requirement | Specification |
|-------------|---------------|
| **Target Accuracy** | 95%+ |
| **Primary Goal** | Multi-task: Murmur detection + Clinical outcome |
| **Deployment (Initial)** | Local Python environment |
| **Deployment (Future)** | REST API (FastAPI) |
| **Hardware** | Google Colab Pro (GPU available) |
| **Inference Input** | Audio file only (no TSV required) |

## Key Constraints

1. **Inference Simplicity**: Model must work with audio-only input during inference
2. **TSV Usage**: Segmentation annotations can be used for training but NOT required at inference
3. **Accuracy Priority**: Optimize for highest accuracy over speed or interpretability
4. **GPU Training**: Leverage Colab Pro GPUs for faster training

## Success Criteria

- [ ] Murmur detection accuracy ≥ 95%
- [ ] Clinical outcome accuracy ≥ 95%
- [ ] Inference works with raw audio files only
- [ ] Model can be loaded and run locally
- [ ] Clear API interface for future web deployment
