"""
Inference Predictor for LightCardiacNet

Provides HeartSoundPredictor class for CPU-only inference.
"""

import os
import time
import json
import torch
from typing import List, Dict, Union, Optional
from datetime import datetime

from .lightcardiacnet import LightCardiacNet, create_model
from .features import extract_features_for_bigru


class HeartSoundPredictor:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, path: str) -> LightCardiacNet:
        checkpoint = torch.load(path, map_location=self.device)
        
        config = checkpoint.get('config', {})
        model = create_model(
            model_type=config.get('model_type', 'ensemble'),
            input_size=config.get('input_size', 39),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(self.device)
    
    def predict(self, audio_path: str, threshold: float = 0.5) -> Dict:
        start_time = time.time()
        
        try:
            features = extract_features_for_bigru(audio_path)
            features = features.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits, (attn1, attn2) = self.model(features)
                probs = torch.sigmoid(logits).squeeze()
            
            heart_prob = probs[0].item()
            murmur_prob = probs[1].item()
            
            heart_sound_present = heart_prob > threshold
            murmur_present = murmur_prob > threshold if heart_sound_present else False
            
            inference_time = (time.time() - start_time) * 1000
            
            return {
                "success": True,
                "prediction": {
                    "heart_sound_present": bool(heart_sound_present),
                    "murmur_present": bool(murmur_present)
                },
                "confidence": {
                    "heart_sound": round(heart_prob, 4),
                    "murmur": round(murmur_prob, 4)
                },
                "inference_time_ms": round(inference_time, 2),
                "metadata": {
                    "model_version": "lightcardiacnet_v1",
                    "processing_date": datetime.now().isoformat(),
                    "audio_file": os.path.basename(audio_path)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "audio_file": audio_path
            }
    
    def predict_patient(
        self,
        audio_files: List[str],
        threshold: float = 0.5
    ) -> Dict:
        start_time = time.time()
        
        predictions = []
        for audio_path in audio_files:
            pred = self.predict(audio_path, threshold)
            predictions.append(pred)
        
        heart_sound = any(
            p['prediction']['heart_sound_present']
            for p in predictions if p['success']
        )
        murmur = any(
            p['prediction']['murmur_present']
            for p in predictions if p['success']
        )
        
        max_heart_conf = max(
            (p['confidence']['heart_sound'] for p in predictions if p['success']),
            default=0.0
        )
        max_murmur_conf = max(
            (p['confidence']['murmur'] for p in predictions if p['success']),
            default=0.0
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "patient_prediction": {
                "heart_sound_present": heart_sound,
                "murmur_present": murmur
            },
            "patient_confidence": {
                "heart_sound": round(max_heart_conf, 4),
                "murmur": round(max_murmur_conf, 4)
            },
            "per_valve_predictions": predictions,
            "total_inference_time_ms": round(total_time, 2),
            "num_valves_analyzed": len(audio_files)
        }
    
    def predict_batch(
        self,
        audio_paths: List[str],
        threshold: float = 0.5
    ) -> List[Dict]:
        return [self.predict(path, threshold) for path in audio_paths]


def load_predictor(model_path: str = 'model/checkpoints/best_model.pt') -> HeartSoundPredictor:
    return HeartSoundPredictor(model_path, device='cpu')
