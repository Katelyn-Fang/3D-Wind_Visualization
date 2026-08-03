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
MODEL_PIPELINE_ROOT = PROJECT_ROOT / "wind_ml_pipeline_12_models"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODEL_PIPELINE_ROOT))
from src.wind_core import (  # noqa: E402
    add_engineered_features,
    modeled_to_absolute_angle,
    vectors_to_angle_deg,
    yaw_to_heading_deg,
)

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

    engineered, _, _ = add_engineered_features(
        pd.DataFrame(rows), attitude_angle_unit="radians"
    )
    features = artifact["feature_columns"]
    latest = engineered[features].tail(1)
    speed = max(float(artifact["speed_model"].predict(latest)[0]), 0.0)
    direction_vector = artifact["direction_model"].predict(latest)
    direction_confidence = float(
        np.clip(np.linalg.norm(np.asarray(direction_vector), axis=1)[0], 0.0, 1.0)
    )
    modeled_direction_deg = vectors_to_angle_deg(np.asarray(direction_vector))
    training_arguments = artifact.get("training_arguments", {})
    direction_target = training_arguments.get("direction_target", "absolute")
    if direction_target == "relative_yaw":
        yaw_heading = yaw_to_heading_deg(
            engineered.tail(1)["Yaw"].to_numpy(),
            "radians",
            training_arguments.get("yaw_transform", "clockwise_from_north"),
        )
    else:
        yaw_heading = None
    direction_from_deg = float(
        modeled_to_absolute_angle(
            modeled_direction_deg,
            direction_target,
            yaw_heading,
        )[0]
    )

    # Meteorological direction is where wind comes from; u/v point where it goes.
    angle = np.deg2rad(direction_from_deg)
    u = -speed * np.sin(angle)
    v = -speed * np.cos(angle)
    return {
        "speed": speed,
        "direction_from_deg": direction_from_deg,
        "direction_confidence": direction_confidence,
        "u": float(u),
        "v": float(v),
        "w": 0.0,
    }
