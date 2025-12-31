# Step 7: Inference API

## Objective
Create a simple inference interface for real-world usage.

## Tasks

### 7.1 Inference Module
```python
class HeartSoundPredictor:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        self.model.eval()
    
    def _load_model(self, path):
        model = LightCardiacNet()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model
    
    def predict(self, audio_path: str) -> dict:
        """
        Predict heart sound and murmur from audio file.
        
        Returns:
            {
                "success": true,
                "prediction": {
                    "heart_sound_present": true,
                    "murmur_present": false
                },
                "confidence": {...},
                "inference_time_ms": 18
            }
        """
        start_time = time.time()
        
        features = extract_features(audio_path)
        
        with torch.no_grad():
            logits, _ = self.model(features.unsqueeze(0))
            probs = torch.sigmoid(logits).squeeze()
        
        heart_prob = probs[0].item()
        murmur_prob = probs[1].item()
        
        heart_sound_present = heart_prob > 0.5
        murmur_present = murmur_prob > 0.5 if heart_sound_present else False
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "prediction": {
                "heart_sound_present": heart_sound_present,
                "murmur_present": murmur_present
            },
            "confidence": {
                "heart_sound": round(heart_prob, 4),
                "murmur": round(murmur_prob, 4)
            },
            "inference_time_ms": round(inference_time, 2)
        }
```

### 7.2 Multi-File Processing
```python
def predict_patient(audio_files: list) -> dict:
    """Process multiple valve recordings for one patient."""
    predictions = []
    
    for audio_path in audio_files:
        pred = predictor.predict(audio_path)
        predictions.append(pred)
    
    # Decision-level fusion
    heart_sound = any(p['prediction']['heart_sound_present'] for p in predictions)
    murmur = any(p['prediction']['murmur_present'] for p in predictions)
    
    return {
        "heart_sound_present": heart_sound,
        "murmur_present": murmur,
        "per_valve_predictions": predictions
    }
```

### 7.3 CLI Interface
```python
# predict.py
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default="models/best_model.pt")
    args = parser.parse_args()
    
    predictor = HeartSoundPredictor(args.model)
    result = predictor.predict(args.audio_path)
    print(json.dumps(result, indent=2))
```

Usage:
```bash
python predict.py sample_audio.wav --model models/lightcardiacnet_pruned.pt
```

## Deliverables
- [ ] `predict.py` - CLI inference tool
- [ ] `model/predictor.py` - Predictor class

## Estimated Time
~2 hours
