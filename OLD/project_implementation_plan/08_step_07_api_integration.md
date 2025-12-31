# Step 7: API Integration (Future)

## Objective
Create a REST API for the trained model using FastAPI, enabling external applications to make predictions.

## Prerequisites
- Step 6 completed (inference pipeline ready)
- Trained model checkpoint available

---

## Implementation Details

### 7.1 FastAPI Application

Create `api/app.py`:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from src.inference.predictor import HeartSoundPredictor

app = FastAPI(
    title="Jivascope Heart Sound Classifier",
    description="API for detecting heart murmurs and clinical outcomes from audio",
    version="1.0.0"
)

# Load model on startup
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    predictor = HeartSoundPredictor('checkpoints/best_model.pt')
    print("Model loaded successfully!")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict heart murmur and clinical outcome from audio file.
    
    Accepts: .wav audio file
    Returns: JSON with predictions and confidence scores
    """
    if not file.filename.endswith('.wav'):
        raise HTTPException(400, "Only .wav files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = predictor.predict(tmp_path)
        return JSONResponse(content=result)
    finally:
        os.unlink(tmp_path)

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch prediction for multiple audio files.
    """
    results = []
    for file in files:
        # Process each file
        pass
    return results
```

### 7.2 Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 7.3 API Documentation

FastAPI auto-generates OpenAPI docs at `/docs` endpoint.

---

## Research Areas
- Model optimization for faster inference (ONNX, TorchScript)
- GPU inference in production
- Rate limiting and authentication

---

## Expected Outcome
- FastAPI application with `/predict` endpoint
- Dockerfile for containerized deployment
- API documentation

---

## Estimated Effort
- **4-6 hours** for API development

---

## Dependencies
- Step 6: Inference pipeline
