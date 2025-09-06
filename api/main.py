from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from edf.evaluate import predict_at
from typing import Dict, Optional, List
from fastapi import FastAPI
from pathlib import Path
from datetime import datetime, timezone
import os

app = FastAPI(title="Electricity Demand Predictor", version="1.0.0")

RUNS: Dict[str, Path] = {
    '2025-08-23_13-32-41': Path('./models/2025-08-23_13-32-41'),
    '2025-09-05_07-07-46': Path('./models/2025-09-05_07-07-46')
}


class PredictionRequest(BaseModel):
    run_id: str = Field(..., description="ID of the run",
                        examples=["2025-09-05_07-07-46"])
    timestamp: Optional[datetime] = Field(
        None, description="Timestamp for the prediction request", examples=["2020-08-22T14:30:00"])


class PredictResponse(BaseModel):
    y_pred: List[float]
    y_true: Optional[List[float]]
    timestamps: List[datetime]


def _resolve_run_dir(run_id: str) -> Path:
    try:
        run_dir = RUNS[run_id]
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Unknown run_id: {run_id}")
    if not run_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"Run directory missing on disk: {run_dir}")
    return run_dir


def _call_predict(run_dir: Path, timestamp: Optional[datetime] = None):
    """
    Helper function to call the predict_at function.
    """
    try:
        predictions, y_true, timestamps = predict_at(
            run_dir, timestamp=timestamp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, detail=f"Model artifacts missing: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Prediction failed unexpectedly") from e

    # Convert to list to serialize with pydantic
    y_pred_list = predictions.tolist() if hasattr(
        predictions, "tolist") else list(predictions)
    y_true_list = None
    if y_true is not None:
        y_true_list = y_true.tolist() if hasattr(y_true, "tolist") else list(y_true)

    ts_list = []
    for ts in timestamps:
        if isinstance(ts, datetime):
            ts_list.append(
                ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))
        else:
            raise HTTPException(
                status_code=500, detail="predict_at returned non-datetime timestamps")

    return y_pred_list, y_true_list, ts_list


@app.get("/")
def read_root():

    return os.listdir('./models')
    return {"message": "Welcome to the Electricity Demand Predictor API"}


@app.post("/predict",)
def predict(request: PredictionRequest):
    run_dir = _resolve_run_dir(request.run_id)
    y_pred, y_true, timestamps = _call_predict(
        run_dir, timestamp=request.timestamp)
    return PredictResponse(y_pred=y_pred, y_true=y_true, timestamps=timestamps)
