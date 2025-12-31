"""
LightCardiacNet Inference CLI

Usage:
    python predict.py audio_file.wav                     # Single file prediction
    python predict.py file1.wav file2.wav --patient      # Multi-valve patient prediction
    python predict.py audio_file.wav --model path/to/model.pt
"""

import os
import sys
import json
import argparse
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.predictor import HeartSoundPredictor


def main():
    parser = argparse.ArgumentParser(description='Heart Sound Prediction CLI')
    parser.add_argument('audio_paths', nargs='+', help='Path(s) to audio file(s)')
    parser.add_argument('--model', type=str, default='model/checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--patient', action='store_true',
                       help='Treat multiple files as patient multi-valve recordings')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    
    audio_files = []
    for path in args.audio_paths:
        if '*' in path:
            audio_files.extend(glob(path))
        else:
            audio_files.append(path)
    
    for f in audio_files:
        if not os.path.exists(f):
            print(f"Error: Audio file not found: {f}")
            sys.exit(1)
    
    if args.verbose:
        print(f"Loading model from: {args.model}")
    
    predictor = HeartSoundPredictor(args.model, device='cpu')
    
    if args.verbose:
        print(f"Processing {len(audio_files)} file(s)...")
        print()
    
    if args.patient and len(audio_files) > 1:
        result = predictor.predict_patient(audio_files, args.threshold)
    elif len(audio_files) == 1:
        result = predictor.predict(audio_files[0], args.threshold)
    else:
        results = predictor.predict_batch(audio_files, args.threshold)
        result = {"predictions": results}
    
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        if args.verbose:
            print(f"Results saved to: {args.output}")
    else:
        print(output_json)
    
    if not args.output and args.verbose:
        print("\n" + "="*40)
        if 'patient_prediction' in result:
            pred = result['patient_prediction']
            print(f"Patient Summary:")
            print(f"  Heart Sound: {'✓ Present' if pred['heart_sound_present'] else '✗ Absent'}")
            print(f"  Murmur: {'✓ Present' if pred['murmur_present'] else '✗ Absent'}")
        elif 'prediction' in result:
            pred = result['prediction']
            print(f"Prediction:")
            print(f"  Heart Sound: {'✓ Present' if pred['heart_sound_present'] else '✗ Absent'}")
            print(f"  Murmur: {'✓ Present' if pred['murmur_present'] else '✗ Absent'}")


if __name__ == "__main__":
    main()
