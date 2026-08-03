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
VALIDATION_PATH = Path(
    os.environ.get("WIND_VALIDATION_PATH", str(PROJECT_ROOT / "test_predictions.csv"))
).expanduser()
PHYSICS_VALIDATION_PATH = Path(
    os.environ.get(
        "WIND_PHYSICS_VALIDATION_PATH",
        str(PROJECT_ROOT / "physics_test_predictions.csv"),
    )
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
_validation_metrics: dict[str, float | int] | None = None
_validation_frame: pd.DataFrame | None = None
_physics_validation_frame: pd.DataFrame | None = None


def get_validation_frame() -> pd.DataFrame:
    global _validation_frame
    if _validation_frame is None:
        if not VALIDATION_PATH.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Validation predictions not found: {VALIDATION_PATH}",
            )
        _validation_frame = pd.read_csv(VALIDATION_PATH).dropna(
            subset=["Wind_speed", "Predicted_wind_speed", "Predicted_wind_angle"]
        ).reset_index(drop=True)
    return _validation_frame


def get_physics_validation_frame() -> pd.DataFrame:
    global _physics_validation_frame
    if _physics_validation_frame is None:
        if not PHYSICS_VALIDATION_PATH.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Physics validation predictions not found: {PHYSICS_VALIDATION_PATH}",
            )
        frame = pd.read_csv(PHYSICS_VALIDATION_PATH)
        if len(frame) != len(get_validation_frame()):
            raise HTTPException(
                status_code=500,
                detail="Physics and ML validation files do not contain the same held-out rows",
            )
        _physics_validation_frame = frame
    return _physics_validation_frame


def wind_components(speed: np.ndarray, angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(angle_deg)
    return -speed * np.sin(angle), -speed * np.cos(angle)


def vector_metrics(pred_u: np.ndarray, pred_v: np.ndarray, true_u: np.ndarray, true_v: np.ndarray) -> dict[str, float]:
    delta_u = pred_u - true_u
    delta_v = pred_v - true_v
    vector_error = np.hypot(delta_u, delta_v)
    predicted_speed = np.hypot(pred_u, pred_v)
    measured_speed = np.hypot(true_u, true_v)
    speed_error = predicted_speed - measured_speed
    dot = pred_u * true_u + pred_v * true_v
    denominator = np.maximum(predicted_speed * measured_speed, 1e-9)
    direction_error = np.rad2deg(np.arccos(np.clip(dot / denominator, -1, 1)))
    reliable = (predicted_speed >= 0.2) & (measured_speed >= 1.0)
    return {
        "vector_mae_mps": float(np.mean(vector_error)),
        "vector_rmse_mps": float(np.sqrt(np.mean(vector_error**2))),
        "speed_mae_mps": float(np.mean(np.abs(speed_error))),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
        "direction_mae_deg": float(np.mean(direction_error[reliable])),
    }


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


@app.get("/validation-metrics")
def validation_metrics() -> dict[str, Any]:
    """Return held-out errors against the measured anemometer wind speed."""
    global _validation_metrics
    if _validation_metrics is not None:
        return _validation_metrics
    frame = get_validation_frame()
    measured = frame["Wind_speed"].to_numpy(dtype=float)
    predicted = frame["Predicted_wind_speed"].to_numpy(dtype=float)
    absolute_error = np.abs(predicted - measured)
    percent_mask = measured >= 0.5
    _validation_metrics = {
        "sample_count": int(len(frame)),
        "measured_mean_mps": float(np.mean(measured)),
        "speed_mae_mps": float(np.mean(absolute_error)),
        "speed_rmse_mps": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "speed_mape_percent": float(
            np.mean(absolute_error[percent_mask] / measured[percent_mask]) * 100
        ),
    }
    return _validation_metrics


@app.get("/comparison-metrics")
def comparison_metrics() -> dict[str, Any]:
    """Compare independent ML and physics predictions on identical test rows."""
    ml = get_validation_frame()
    physics = get_physics_validation_frame()
    measured_u, measured_v = wind_components(
        ml["Wind_speed"].to_numpy(float), ml["Wind_angle"].to_numpy(float)
    )
    ml_u, ml_v = wind_components(
        ml["Predicted_wind_speed"].to_numpy(float),
        ml["Predicted_wind_angle"].to_numpy(float),
    )
    return {
        "sample_count": int(len(ml)),
        "held_out_flight_count": int(ml["Flight_ID"].nunique()),
        "ml": vector_metrics(ml_u, ml_v, measured_u, measured_v),
        "physics": vector_metrics(
            physics["Physics_u"].to_numpy(float),
            physics["Physics_v"].to_numpy(float),
            measured_u,
            measured_v,
        ),
        "comparison_is_independent": True,
    }


@app.get("/validation-sample")
def validation_sample(index: int = 0) -> dict[str, Any]:
    """Return one recorded test row with matched prediction and measurement."""
    frame = get_validation_frame()
    if index < 0 or index >= len(frame):
        raise HTTPException(status_code=400, detail=f"index must be 0..{len(frame) - 1}")
    row = frame.iloc[index]
    physics_row = get_physics_validation_frame().iloc[index]
    measured = float(row["Wind_speed"])
    predicted = max(float(row["Predicted_wind_speed"]), 0.0)
    angle = float(row["Predicted_wind_angle"])
    radians = np.deg2rad(angle)
    absolute_error = abs(predicted - measured)
    return {
        "index": index,
        "sample_count": int(len(frame)),
        "flight_id": str(row.get("Flight_ID", "")),
        "measured_speed": measured,
        "predicted_speed": predicted,
        "speed_error": absolute_error,
        "speed_error_percent": absolute_error / measured * 100 if measured >= 0.1 else None,
        "direction_from_deg": angle,
        "direction_confidence": 1.0,
        "speed": predicted,
        "u": float(-predicted * np.sin(radians)),
        "v": float(-predicted * np.cos(radians)),
        "w": 0.0,
        "measured_u": float(-measured * np.sin(np.deg2rad(float(row["Wind_angle"])))),
        "measured_v": float(-measured * np.cos(np.deg2rad(float(row["Wind_angle"])))),
        "physics_u": float(physics_row["Physics_u"]),
        "physics_v": float(physics_row["Physics_v"]),
        "physics_speed": float(physics_row["Physics_speed"]),
        "physics_vector_error": float(physics_row["Physics_vector_error"]),
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
