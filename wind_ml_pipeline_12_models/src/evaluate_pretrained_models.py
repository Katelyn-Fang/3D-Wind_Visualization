"""Evaluate saved wind models on a new standardized dataset without fitting."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from .neural_models import build_network
from .wind_core import (
    add_engineered_features,
    circular_difference_deg,
    load_standardized_data,
    modeled_to_absolute_angle,
    vectors_to_angle_and_confidence,
    yaw_to_heading_deg,
)


def baseline_artifacts(results_root: Path) -> list[Path]:
    """Return the 8 tabular and 4 neural baseline artifacts in display order."""
    cpu = sorted((results_root / "cpu").glob("*/wind_model.joblib"))
    gpu = sorted((results_root / "gpu").glob("*/wind_model.pt"))
    return cpu + gpu


def _metrics(frame: pd.DataFrame, model: str, artifact: Path) -> dict:
    speed_error = frame["Predicted_wind_speed"] - frame["Wind_speed"]
    angle_error = circular_difference_deg(
        frame["Wind_angle"].to_numpy(), frame["Predicted_wind_angle"].to_numpy()
    )
    reliable = frame["Wind_speed"].to_numpy() >= 1.0
    per_flight = frame.assign(
        _speed_abs=speed_error.abs(), _direction_abs=np.abs(angle_error)
    ).groupby("_Group_ID", observed=True)[["_speed_abs", "_direction_abs"]].mean()
    return {
        "model": model,
        "rows_evaluated": len(frame),
        "flights_evaluated": frame["_Group_ID"].nunique(),
        "speed_mae_mps": float(np.mean(np.abs(speed_error))),
        "speed_rmse_mps": float(np.sqrt(mean_squared_error(frame["Wind_speed"], frame["Predicted_wind_speed"]))),
        "speed_r2": float(r2_score(frame["Wind_speed"], frame["Predicted_wind_speed"])),
        "direction_circular_mae_deg": float(np.mean(np.abs(angle_error))),
        "direction_within_30deg_fraction": float(np.mean(np.abs(angle_error) <= 30.0)),
        "direction_speed_ge_1_mps_mae_deg": (
            float(np.mean(np.abs(angle_error[reliable]))) if reliable.any() else np.nan
        ),
        "flight_balanced_speed_mae_mps": float(per_flight["_speed_abs"].mean()),
        "flight_balanced_direction_mae_deg": float(per_flight["_direction_abs"].mean()),
        "artifact": str(artifact),
    }


def _per_flight_metrics(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Calculate interpretable error metrics separately for every flight."""
    working = frame.copy()
    working["speed_error_mps"] = (
        working["Predicted_wind_speed"] - working["Wind_speed"]
    )
    working["speed_absolute_error_mps"] = working["speed_error_mps"].abs()
    working["direction_error_deg"] = circular_difference_deg(
        working["Wind_angle"].to_numpy(),
        working["Predicted_wind_angle"].to_numpy(),
    )
    working["direction_absolute_error_deg"] = working["direction_error_deg"].abs()
    working["direction_reliable_absolute_error_deg"] = (
        working["direction_absolute_error_deg"].where(working["Wind_speed"] >= 1.0)
    )

    rows = []
    for (group_id, flight_id), part in working.groupby(
        ["_Group_ID", "Flight_ID"], sort=True, observed=True
    ):
        speed_error = part["speed_error_mps"].to_numpy(dtype=float)
        direction_error = part["direction_error_deg"].to_numpy(dtype=float)
        reliable_direction = part["direction_reliable_absolute_error_deg"].dropna()
        rows.append(
            {
                "model": model,
                "_Group_ID": group_id,
                "Flight_ID": flight_id,
                "rows_evaluated": len(part),
                "mean_true_speed_mps": part["Wind_speed"].mean(),
                "mean_true_direction_deg": part["Wind_angle"].mean(),
                "speed_bias_mps": float(np.mean(speed_error)),
                "speed_mae_mps": float(np.mean(np.abs(speed_error))),
                "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
                "direction_bias_deg": float(np.mean(direction_error)),
                "direction_circular_mae_deg": float(np.mean(np.abs(direction_error))),
                "direction_circular_rmse_deg": float(
                    np.sqrt(np.mean(direction_error**2))
                ),
                "direction_within_30deg_fraction": float(
                    np.mean(np.abs(direction_error) <= 30.0)
                ),
                "direction_speed_ge_1_mps_mae_deg": (
                    float(reliable_direction.mean())
                    if len(reliable_direction)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _absolute_angle(modeled_angle, engineered, indices, target_mode, yaw_transform):
    if target_mode == "absolute":
        return modeled_angle
    heading = yaw_to_heading_deg(
        engineered.iloc[indices]["Yaw"].to_numpy(),
        "radians",
        yaw_transform,
    )
    return modeled_to_absolute_angle(modeled_angle, target_mode, heading)


def _evaluate_tabular(artifact_path: Path, engineered: pd.DataFrame):
    artifact = joblib.load(artifact_path, mmap_mode="r")
    features = artifact["feature_columns"]
    missing = sorted(set(features) - set(engineered.columns))
    if missing:
        raise ValueError(f"Engineered data lacks model features: {missing}")
    X = engineered[features]
    speed = np.maximum(np.asarray(artifact["speed_model"].predict(X), dtype=float), 0.0)
    vectors = np.asarray(artifact["direction_model"].predict(X), dtype=float)
    modeled, _ = vectors_to_angle_and_confidence(vectors)
    args = artifact.get("training_arguments", {})
    angle = _absolute_angle(
        modeled,
        engineered,
        np.arange(len(engineered)),
        args.get("direction_target", "absolute"),
        args.get("yaw_transform", "clockwise_from_north"),
    )
    return artifact.get("model_name", artifact_path.parent.name), np.arange(len(engineered)), speed, angle


def _evaluate_neural(artifact_path: Path, engineered: pd.DataFrame, batch_size=2048):
    import torch

    checkpoint = torch.load(artifact_path, map_location="cpu", weights_only=False)
    features = checkpoint["feature_columns"]
    raw = engineered[features].to_numpy(dtype=float)
    state = checkpoint["preprocessor"]
    statistics = np.asarray(state["imputer_statistics"], dtype=float)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    raw = np.where(np.isnan(raw), statistics[None, :], raw)
    matrix = ((raw - np.asarray(state["scaler_mean"])) / np.asarray(state["scaler_scale"])).astype(np.float32)

    config = checkpoint["model_config"]
    model_name = checkpoint["model_name"]
    args = SimpleNamespace(**config)
    model = build_network(model_name, len(features), args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    sequence_length = 1 if model_name == "mlp" else int(config["sequence_length"])
    endpoints = []
    for _, part in engineered.groupby("_Group_ID", sort=False, observed=True):
        idx = part.index.to_numpy(dtype=np.int64)
        endpoints.extend(idx if model_name == "mlp" else idx[sequence_length - 1 :])
    endpoints = np.asarray(endpoints, dtype=np.int64)

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(endpoints), batch_size):
            batch_endpoints = endpoints[start : start + batch_size]
            if model_name == "mlp":
                batch = matrix[batch_endpoints]
            else:
                batch = np.stack(
                    [matrix[i - sequence_length + 1 : i + 1] for i in batch_endpoints]
                )
            outputs.append(model(torch.from_numpy(batch)).numpy())
    output = np.concatenate(outputs)
    scaling = checkpoint["target_scaling"]
    speed = np.maximum(output[:, 0] * scaling["speed_std"] + scaling["speed_mean"], 0.0)
    modeled, _ = vectors_to_angle_and_confidence(output[:, 1:3])
    angle = _absolute_angle(
        modeled,
        engineered,
        endpoints,
        config.get("direction_target_mode", "absolute"),
        config.get("yaw_transform", "clockwise_from_north"),
    )
    return model_name, endpoints, speed, angle


def evaluate_all(
    data_path: Path,
    results_root: Path,
    output_dir: Path,
    *,
    max_rows_per_flight: int | None = None,
    save_predictions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every available baseline artifact and save comparable results."""
    data = load_standardized_data(data_path)
    if max_rows_per_flight:
        group = data["Source_dataset"].astype(str) + "::" + data["Flight_ID"].astype(str)
        data = data[group.groupby(group).cumcount() < max_rows_per_flight].copy()
    engineered, _, metadata = add_engineered_features(data, attitude_angle_unit="radians")
    artifacts = baseline_artifacts(results_root)
    if not artifacts:
        raise FileNotFoundError(f"No saved model artifacts found below {results_root}")

    rows, failures, per_flight_tables = [], [], []
    for artifact in artifacts:
        try:
            if artifact.suffix == ".joblib":
                name, indices, speed, angle = _evaluate_tabular(artifact, engineered)
            else:
                name, indices, speed, angle = _evaluate_neural(artifact, engineered)
            prediction = engineered.iloc[indices][
                ["_Group_ID", "Flight_ID", "Wind_speed", "Wind_angle"]
            ].copy()
            prediction["model"] = name
            prediction["Predicted_wind_speed"] = speed
            prediction["Predicted_wind_angle"] = angle
            rows.append(_metrics(prediction, name, artifact))
            per_flight_tables.append(_per_flight_metrics(prediction, name))
            if save_predictions:
                prediction_dir = output_dir / "predictions"
                prediction_dir.mkdir(parents=True, exist_ok=True)
                prediction.to_csv(prediction_dir / f"{name}.csv", index=False)
        except Exception as exc:
            failures.append({"artifact": str(artifact), "error": f"{type(exc).__name__}: {exc}"})

    comparison = pd.DataFrame(rows).sort_values(
        ["speed_mae_mps", "direction_circular_mae_deg"], ignore_index=True
    )
    failure_frame = pd.DataFrame(failures, columns=["artifact", "error"])
    per_flight = (
        pd.concat(per_flight_tables, ignore_index=True)
        if per_flight_tables
        else pd.DataFrame()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    failure_frame.to_csv(output_dir / "model_failures.csv", index=False)
    per_flight.to_csv(output_dir / "per_flight_model_comparison.csv", index=False)
    with (output_dir / "evaluation_info.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "data": str(data_path.resolve()),
                "results_root": str(results_root.resolve()),
                "rows_after_optional_limit": len(data),
                "max_rows_per_flight": max_rows_per_flight,
                "predictions_saved": save_predictions,
                "attitude_angle_unit": metadata["attitude_angle_unit_resolved"],
                "retrained": False,
            },
            handle,
            indent=2,
        )
    return comparison, failure_frame
