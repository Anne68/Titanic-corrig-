import os
import joblib
from fastapi import FastAPI, HTTPException
from typing import List
from src.schemas import TitanicPayload
from src.logger import logger
from src.monitoring import setup_instrumentation

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
app = FastAPI(title="Titanic API", version="1.0.0")

model = None

@app.on_event("startup")
def _startup():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"[STARTUP] Model loaded from {MODEL_PATH}")
    else:
        logger.warning(f"[STARTUP] Model NOT found at {MODEL_PATH}")
    setup_instrumentation(app)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "model_path": MODEL_PATH}

@app.post("/predict")
async def predict(payloads: List[TitanicPayload]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it first.")
    rows = [p.dict() for p in payloads]
    import pandas as pd
    X = pd.DataFrame(rows)
    try:
        preds = model.predict(X).tolist()
        probs = model.predict_proba(X)[:, 1].tolist()
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))
    return {"predictions": preds, "proba_survive": probs}
