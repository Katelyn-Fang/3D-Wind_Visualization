"""Local inference API for the trained wind_model.joblib artifact."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from train_wind_model_v2_1 import add_engineered_features, vectors_to_angle_deg  # noqa: E402

MODEL_PATH = Path(
    os.environ.get("WIND_MODEL_PATH", str(Path.home() / "Downloads" / "wind_model.joblib"))
).expanduser()

app = FastAPI(title="Drone wind model API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Telemetry(BaseModel):
    session_id: str = "vite-simulator"
    elapsed_s: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    battery_v: float = 15.2
    battery_c: float = 4.0


_artifact: dict[str, Any] | None = None
_load_lock = threading.Lock()
_history: dict[str, list[dict[str, Any]]] = {}


def get_artifact() -> dict[str, Any]:
    global _artifact
    if _artifact is None:
        with _load_lock:
            if _artifact is None:
                if not MODEL_PATH.is_file():
                    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
                _artifact = joblib.load(MODEL_PATH)
    return _artifact


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": MODEL_PATH.is_file(),
        "model_path": str(MODEL_PATH),
        "model_loaded": _artifact is not None,
        "model_size_bytes": MODEL_PATH.stat().st_size if MODEL_PATH.is_file() else None,
    }


@app.post("/predict")
def predict(sample: Telemetry) -> dict[str, float]:
    try:
        artifact = get_artifact()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    rows = _history.setdefault(sample.session_id, [])
    rows.append({
        "Source_dataset": "vite-simulator",
        "Flight_ID": sample.session_id,
        "Elapsed_s": sample.elapsed_s,
        "X": sample.x,
        "Y": sample.y,
        "Z": sample.z,
        "Roll": sample.roll,
        "Pitch": sample.pitch,
        "Yaw": sample.yaw,
        "Battery_V": sample.battery_v,
        "Battery_C": sample.battery_c,
    })
    del rows[:-120]

    engineered, _, _ = add_engineered_features(pd.DataFrame(rows))
    features = artifact["feature_columns"]
    latest = engineered[features].tail(1)
    speed = max(float(artifact["speed_model"].predict(latest)[0]), 0.0)
    direction_vector = artifact["direction_model"].predict(latest)
    direction_from_deg = float(vectors_to_angle_deg(np.asarray(direction_vector))[0])

    # Meteorological direction is where wind comes from; u/v point where it goes.
    angle = np.deg2rad(direction_from_deg)
    u = -speed * np.sin(angle)
    v = -speed * np.cos(angle)
    return {
        "speed": speed,
        "direction_from_deg": direction_from_deg,
        "u": float(u),
        "v": float(v),
        "w": 0.0,
    }
