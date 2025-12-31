"""
Jivascope Single Audio Test Script

Test a specific model with a specific audio file and get predictions.

Usage:
    python test_model.py --model checkpoints/best_model.pt --audio path/to/audio.wav
    python test_model.py --model checkpoints/Jivascope_checkpoints_best_model.pt --audio path/to/audio.wav
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from src.models import HeartSoundClassifier, create_model
from src.data.preprocessing import (
    load_audio,
    normalize_audio,
    pad_or_truncate,
    apply_bandpass_filter,
    create_multichannel_spectrogram,
    SAMPLE_RATE,
    TARGET_DURATION
)


MURMUR_LABELS = ['Absent', 'Present']
OUTCOME_LABELS = ['Normal', 'Abnormal']


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cpu'):
    """
    Load a trained model from checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'backbone_name' in checkpoint:
        backbone_name = checkpoint['backbone_name']
    elif 'config' in checkpoint and hasattr(checkpoint['config'], 'model'):
        backbone_name = checkpoint['config'].model.backbone
    else:
        file_size_mb = Path(checkpoint_path).stat().st_size / (1024 * 1024)
        backbone_name = 'resnet18' if file_size_mb > 100 else 'efficientnet_b0'
        print(f"  ⚠️ Backbone not in checkpoint, inferring '{backbone_name}' from file size ({file_size_mb:.1f}MB)")
    
    model = create_model(
        backbone_name=backbone_name,
        pretrained=False,
        device=device
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    info = {
        'backbone': backbone_name,
        'epoch': checkpoint.get('epoch', 'N/A'),
        'best_val_loss': checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'N/A')),
    }
    
    return model, info


def preprocess_audio(audio_path: str, sample_rate: int = SAMPLE_RATE, target_duration: float = TARGET_DURATION):
    """
    Preprocess a single audio file for model inference.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate
        target_duration: Target duration in seconds
        
    Returns:
        Preprocessed spectrogram tensor
    """
    target_length = int(target_duration * sample_rate)
    
    audio = load_audio(audio_path, sample_rate)
    audio = apply_bandpass_filter(audio, sample_rate)
    audio = normalize_audio(audio)
    audio = pad_or_truncate(audio, target_length)
    
    spectrogram = create_multichannel_spectrogram(
        audio,
        sample_rate,
        target_height=128,
        target_width=313
    )
    
    spectrogram = torch.from_numpy(spectrogram).float()
    spectrogram = spectrogram.unsqueeze(0)
    
    return spectrogram


def predict_single_audio(model: HeartSoundClassifier, audio_path: str, device: str = 'cpu'):
    """
    Get prediction for a single audio file.
    
    Args:
        model: Trained model
        audio_path: Path to the audio file
        device: Device to run on
        
    Returns:
        Dictionary with predictions and probabilities
    """
    spectrogram = preprocess_audio(audio_path)
    spectrogram = spectrogram.to(device)
    
    with torch.no_grad():
        probs = model.get_probabilities(spectrogram)
        preds = model.get_predictions(spectrogram)
        
        murmur_prob = probs['murmur'][0].cpu().numpy()
        outcome_prob = probs['outcome'][0].cpu().numpy()
        murmur_pred = preds['murmur'][0].cpu().item()
        outcome_pred = preds['outcome'][0].cpu().item()
    
    return {
        'murmur_prediction': MURMUR_LABELS[murmur_pred],
        'murmur_confidence': murmur_prob[murmur_pred] * 100,
        'murmur_probabilities': {
            'Absent': murmur_prob[0] * 100,
            'Present': murmur_prob[1] * 100
        },
        'outcome_prediction': OUTCOME_LABELS[outcome_pred],
        'outcome_confidence': outcome_prob[outcome_pred] * 100,
        'outcome_probabilities': {
            'Normal': outcome_prob[0] * 100,
            'Abnormal': outcome_prob[1] * 100
        }
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Test Jivascope model on a single audio file')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint (e.g., checkpoints/best_model.pt)')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to the audio file (.wav)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("🫀 JIVASCOPE - Single Audio Test")
    print("=" * 60)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        sys.exit(1)
    
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"\n❌ Error: Audio file not found at {audio_path}")
        sys.exit(1)
    
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    print(f"\n📥 Loading model: {model_path.name}")
    model, info = load_model_from_checkpoint(str(model_path), device)
    print(f"   Backbone: {info['backbone']}")
    print(f"   Epoch:    {info['epoch']}")
    
    print(f"\n🎵 Processing audio: {audio_path.name}")
    
    result = predict_single_audio(model, str(audio_path), device)
    
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULTS")
    print("=" * 60)
    
    print(f"\n🔊 Murmur Detection:")
    print(f"   Prediction:  {result['murmur_prediction']}")
    print(f"   Confidence:  {result['murmur_confidence']:.2f}%")
    print(f"   Probabilities:")
    print(f"     • Absent:  {result['murmur_probabilities']['Absent']:.2f}%")
    print(f"     • Present: {result['murmur_probabilities']['Present']:.2f}%")

    print(f"\n💓 Outcome Classification:")
    print(f"   Prediction:  {result['outcome_prediction']}")
    print(f"   Confidence:  {result['outcome_confidence']:.2f}%")
    print(f"   Probabilities:")
    print(f"     • Normal:   {result['outcome_probabilities']['Normal']:.2f}%")
    print(f"     • Abnormal: {result['outcome_probabilities']['Abnormal']:.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ Test Complete!")
    print("=" * 60 + "\n")
    
    return result


if __name__ == "__main__":
    main()
