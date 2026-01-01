"""
AST Prediction Script for Heart Sound Classification.

Usage:
    python predict.py --audio path/to/audio.wav
    python predict.py --audio path/to/audio.wav --model checkpoints/best_model.pt
"""

import os
import sys
import argparse
import json
import torch
import time
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ast_model import create_model
from model.dataset import extract_ast_features, AST_SAMPLE_RATE, AST_TARGET_DURATION
from common.audio import load_audio, apply_bandpass_filter, normalize_audio, pad_or_truncate


class ASTPredictor:
    """Predictor class for AST heart sound classification."""
    
    def __init__(
        self,
        model_path: str,
        device: str = None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        
        config = checkpoint.get('config', {})
        self.model = create_model(
            model_type=config.get('model_type', 'pretrained'),
            pretrained_model=config.get('pretrained_model', 'MIT/ast-finetuned-audioset-10-10-0.4593'),
            num_classes=config.get('num_classes', 2)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded successfully on {device}")
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        audio = load_audio(audio_path, target_sr=4000)
        audio = apply_bandpass_filter(audio, sr=4000)
        audio = normalize_audio(audio)
        
        target_length_4k = int(AST_TARGET_DURATION * 4000)
        audio = pad_or_truncate(audio, target_length_4k)
        
        audio_16k = librosa.resample(audio, orig_sr=4000, target_sr=AST_SAMPLE_RATE)
        features = extract_ast_features(audio_16k, sr=AST_SAMPLE_RATE)
        
        return features.unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, audio_path: str) -> dict:
        """Run prediction on audio file."""
        start_time = time.time()
        
        features = self.preprocess_audio(audio_path)
        features = features.to(self.device)
        
        logits = self.model(features)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        inference_time = (time.time() - start_time) * 1000
        
        heart_sound_present = bool(probs[0] > 0.5)
        murmur_present = bool(probs[1] > 0.5) if heart_sound_present else False
        
        result = {
            "success": True,
            "prediction": {
                "heart_sound_present": heart_sound_present,
                "murmur_present": murmur_present
            },
            "confidence": {
                "heart_sound": float(probs[0]),
                "murmur": float(probs[1])
            },
            "inference_time_ms": round(inference_time, 2),
            "metadata": {
                "model_type": "ast",
                "model_version": "ast_v1",
                "audio_file": os.path.basename(audio_path)
            }
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description='AST Heart Sound Prediction')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None)
    args = parser.parse_args()
    
    if args.model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model = os.path.join(script_dir, 'model', 'checkpoints', 'best_model.pt')
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    predictor = ASTPredictor(args.model, args.device)
    result = predictor.predict(args.audio)
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    main()
