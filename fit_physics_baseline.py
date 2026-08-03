#!/usr/bin/env python3
"""Fit an independent, identified linear-dynamics wind baseline.

The model follows the system-identification approach described by
Gonzalez-Rocha et al. (Sensors 2020, 20, 1341): wind is treated as an external
disturbance observed through small perturbations of the aircraft state.  It is
not connected to, initialized by, or corrected by the trained ML model.

The held-out flight IDs are read from test_predictions.csv.  Those flights are
excluded from fitting and are used only for the exported comparison results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "roll_rad",
    "pitch_rad",
    "body_velocity_forward_mps",
    "body_velocity_right_mps",
    "vertical_velocity_mps",
    "body_acceleration_x_mps2",
    "body_acceleration_y_mps2",
    "vertical_specific_force_mps2",
    "angular_rate_x_rps",
    "angular_rate_y_rps",
    "angular_rate_z_rps",
    "tilt_squared_rad2",
]


def quaternion_to_euler(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Return aerospace roll, pitch and yaw from x/y/z/w quaternions."""
    x = frame["orientation_x"].to_numpy(float)
    y = frame["orientation_y"].to_numpy(float)
    z = frame["orientation_z"].to_numpy(float)
    w = frame["orientation_w"].to_numpy(float)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    roll, pitch, yaw = quaternion_to_euler(frame)
    vx = frame["velocity_x"].to_numpy(float)
    vy = frame["velocity_y"].to_numpy(float)
    # Rotate horizontal ground velocity into the aircraft heading frame.
    forward = np.cos(yaw) * vx + np.sin(yaw) * vy
    right = -np.sin(yaw) * vx + np.cos(yaw) * vy
    matrix = np.column_stack(
        [
            roll,
            pitch,
            forward,
            right,
            frame["velocity_z"].to_numpy(float),
            frame["linear_acceleration_x"].to_numpy(float),
            frame["linear_acceleration_y"].to_numpy(float),
            frame["linear_acceleration_z"].to_numpy(float) + 9.80665,
            frame["angular_x"].to_numpy(float),
            frame["angular_y"].to_numpy(float),
            frame["angular_z"].to_numpy(float),
            roll * roll + pitch * pitch,
        ]
    )
    return matrix, yaw


def measured_body_wind(frame: pd.DataFrame, yaw: np.ndarray) -> np.ndarray:
    speed = frame["wind_speed"].to_numpy(float)
    angle = np.deg2rad(frame["wind_angle"].to_numpy(float))
    # Meteorological angle is where wind comes from; u/v point where it goes.
    east = -speed * np.sin(angle)
    north = -speed * np.cos(angle)
    forward = np.cos(yaw) * east + np.sin(yaw) * north
    right = -np.sin(yaw) * east + np.cos(yaw) * north
    return np.column_stack([forward, right])


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[scale < 1e-8] = 1
    z = (x - mean) / scale
    augmented = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(augmented.shape[1]) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict(model: dict[str, np.ndarray], x: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    body = np.column_stack([np.ones(len(z)), z]) @ model["coefficients"]
    east = np.cos(yaw) * body[:, 0] - np.sin(yaw) * body[:, 1]
    north = np.sin(yaw) * body[:, 0] + np.cos(yaw) * body[:, 1]
    return np.column_stack([east, north])


def metrics(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float | int]:
    vector_error = np.linalg.norm(predicted - measured, axis=1)
    predicted_speed = np.linalg.norm(predicted, axis=1)
    measured_speed = np.linalg.norm(measured, axis=1)
    speed_error = predicted_speed - measured_speed
    dot = np.sum(predicted * measured, axis=1)
    denom = np.maximum(predicted_speed * measured_speed, 1e-9)
    direction_error = np.rad2deg(np.arccos(np.clip(dot / denom, -1, 1)))
    reliable = (predicted_speed >= 0.2) & (measured_speed >= 1.0)
    return {
        "sample_count": int(len(predicted)),
        "vector_mae_mps": float(np.mean(vector_error)),
        "vector_rmse_mps": float(np.sqrt(np.mean(vector_error**2))),
        "speed_mae_mps": float(np.mean(np.abs(speed_error))),
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
        "direction_mae_deg": float(np.mean(direction_error[reliable])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flights", type=Path, required=True)
    parser.add_argument("--ml-predictions", type=Path, default=Path("test_predictions.csv"))
    parser.add_argument("--output-model", type=Path, default=Path("3DModel/public/data/physics_baseline.json"))
    parser.add_argument("--output-test", type=Path, default=Path("physics_test_predictions.csv"))
    parser.add_argument("--alpha", type=float, default=25.0)
    args = parser.parse_args()

    ml_test = pd.read_csv(args.ml_predictions)
    test_flights = set(pd.to_numeric(ml_test["Flight_ID"]).astype(int))
    required = [
        "flight", "wind_speed", "wind_angle", "position_x", "position_y", "position_z",
        "orientation_x", "orientation_y", "orientation_z", "orientation_w",
        "velocity_x", "velocity_y", "velocity_z", "angular_x", "angular_y", "angular_z",
        "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
    ]
    raw = pd.read_csv(args.flights, usecols=required, low_memory=False).dropna()
    train = raw.loc[~raw["flight"].isin(test_flights)].copy()
    test = raw.loc[raw["flight"].isin(test_flights)].copy()

    x_train, yaw_train = design_matrix(train)
    y_train = measured_body_wind(train, yaw_train)
    model = ridge_fit(x_train, y_train, args.alpha)

    x_test, yaw_test = design_matrix(test)
    predicted = predict(model, x_test, yaw_test)
    measured_body = measured_body_wind(test, yaw_test)
    measured_world = np.column_stack(
        [
            np.cos(yaw_test) * measured_body[:, 0] - np.sin(yaw_test) * measured_body[:, 1],
            np.sin(yaw_test) * measured_body[:, 0] + np.cos(yaw_test) * measured_body[:, 1],
        ]
    )
    result_metrics = metrics(predicted, measured_world)

    artifact = {
        "model": "identified-linear-wind-disturbance-baseline",
        "paper": "https://doi.org/10.3390/s20051341",
        "validity": "hover and steady ascent; small perturbations about equilibrium",
        "feature_names": FEATURE_NAMES,
        "feature_mean": model["mean"].tolist(),
        "feature_scale": model["scale"].tolist(),
        "intercept_body_wind_mps": model["coefficients"][0].tolist(),
        "coefficients_body_wind": model["coefficients"][1:].tolist(),
        "ridge_alpha": args.alpha,
        "training_flight_count": int(train["flight"].nunique()),
        "training_sample_count": int(len(train)),
        "held_out_flight_count": int(test["flight"].nunique()),
        "held_out_metrics": result_metrics,
    }
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(json.dumps(artifact, indent=2) + "\n")

    export = test[["flight", "position_x", "position_y", "position_z", "wind_speed", "wind_angle"]].copy()
    export["Physics_u"] = predicted[:, 0]
    export["Physics_v"] = predicted[:, 1]
    export["Physics_speed"] = np.linalg.norm(predicted, axis=1)
    export["Physics_vector_error"] = np.linalg.norm(predicted - measured_world, axis=1)
    # Reorder to exactly match test_predictions.csv so the UI can compare both
    # estimators on the same slider index. Rounded coordinates plus occurrence
    # number safely handle repeated hover positions.
    def add_join_keys(frame: pd.DataFrame, flight: str, x: str, y: str, z: str) -> pd.DataFrame:
        keyed = frame.copy()
        keyed["_flight_key"] = pd.to_numeric(keyed[flight]).astype(int)
        for source, target in ((x, "_x_key"), (y, "_y_key"), (z, "_z_key")):
            keyed[target] = pd.to_numeric(keyed[source]).round(7)
        group = ["_flight_key", "_x_key", "_y_key", "_z_key"]
        keyed["_occurrence"] = keyed.groupby(group, sort=False).cumcount()
        return keyed

    left = add_join_keys(ml_test.reset_index(names="_ml_index"), "Flight_ID", "X", "Y", "Z")
    right = add_join_keys(export, "flight", "position_x", "position_y", "position_z")
    keys = ["_flight_key", "_x_key", "_y_key", "_z_key", "_occurrence"]
    aligned = left[["_ml_index", *keys]].merge(right, on=keys, how="left", validate="one_to_one")
    if aligned["Physics_speed"].isna().any():
        raise RuntimeError("Could not align every physics result with the held-out ML rows")
    aligned.sort_values("_ml_index")[
        ["flight", "wind_speed", "wind_angle", "Physics_u", "Physics_v", "Physics_speed", "Physics_vector_error"]
    ].to_csv(args.output_test, index=False)
    print(json.dumps(result_metrics, indent=2))


if __name__ == "__main__":
    main()
