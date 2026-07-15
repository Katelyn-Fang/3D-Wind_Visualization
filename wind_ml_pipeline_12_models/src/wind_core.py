"""Shared, leakage-aware utilities for the BU RISE wind-model benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

EARTH_RADIUS_M = 6_371_000.0
BASE_FEATURES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Battery_V", "Battery_C"]
TARGET_COLUMNS = ["Wind_speed", "Wind_angle"]


def circular_difference_deg(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Signed shortest angular difference, predicted minus actual, in [-180, 180)."""
    return (np.asarray(predicted) - np.asarray(actual) + 180.0) % 360.0 - 180.0


def vectors_to_angle_deg(vectors: np.ndarray) -> np.ndarray:
    """Convert columns [sin(angle), cos(angle)] to angles in [0, 360)."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError(f"Expected direction vectors with shape (n, 2); got {vectors.shape}.")
    return (np.rad2deg(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360.0) % 360.0


def vectors_to_angle_and_confidence(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert [sin, cos] outputs to direction and vector-magnitude confidence."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2:
        raise ValueError(f"Expected direction vectors with shape (n, 2); got {vectors.shape}.")
    confidence = np.linalg.norm(vectors, axis=1)
    return vectors_to_angle_deg(vectors), confidence


def direction_training_weights(
    wind_speed: np.ndarray,
    groups: Iterable[str],
    minimum_speed: float = 1.0,
    calm_weight_floor: float = 0.05,
) -> np.ndarray:
    """Combine equal-flight weighting with smooth low-wind down-weighting."""
    speed = np.asarray(wind_speed, dtype=float)
    if minimum_speed < 0:
        raise ValueError("minimum_speed cannot be negative.")
    if not 0.0 < calm_weight_floor <= 1.0:
        raise ValueError("calm_weight_floor must be in (0, 1].")
    flight_weight = equal_flight_weights(groups)
    if minimum_speed == 0:
        reliability = np.ones_like(speed)
    else:
        reliability = np.clip(speed / minimum_speed, calm_weight_floor, 1.0)
    weights = flight_weight * reliability
    return weights / np.mean(weights)


def yaw_to_heading_deg(
    yaw: np.ndarray,
    attitude_angle_unit: str,
    yaw_transform: str,
) -> np.ndarray:
    """Convert yaw values into a clockwise-from-north heading in degrees."""
    if attitude_angle_unit not in {"radians", "degrees"}:
        raise ValueError("attitude_angle_unit must already be resolved to radians or degrees.")
    yaw_deg = np.rad2deg(yaw) if attitude_angle_unit == "radians" else np.asarray(yaw, dtype=float)
    transforms = {
        "clockwise_from_north": yaw_deg,
        "counterclockwise_from_north": -yaw_deg,
        "ccw_from_east_to_heading": 90.0 - yaw_deg,
        "cw_from_east_to_heading": 90.0 + yaw_deg,
    }
    if yaw_transform not in transforms:
        raise ValueError(f"Unknown yaw transform: {yaw_transform}")
    return np.asarray(transforms[yaw_transform], dtype=float) % 360.0


def build_direction_targets(
    wind_angle_deg: np.ndarray,
    *,
    target_mode: str = "absolute",
    yaw: np.ndarray | None = None,
    attitude_angle_unit: str | None = None,
    yaw_transform: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return [sin, cos] targets, modeled angles, and optional yaw headings."""
    wind_angle_deg = np.asarray(wind_angle_deg, dtype=float) % 360.0
    yaw_heading = None
    if target_mode == "absolute":
        modeled_angle = wind_angle_deg
    elif target_mode == "relative_yaw":
        if yaw is None or attitude_angle_unit is None or yaw_transform is None:
            raise ValueError(
                "relative_yaw requires yaw, a resolved attitude unit, and a verified yaw transform."
            )
        yaw_heading = yaw_to_heading_deg(yaw, attitude_angle_unit, yaw_transform)
        modeled_angle = (wind_angle_deg - yaw_heading) % 360.0
    else:
        raise ValueError(f"Unknown direction target mode: {target_mode}")
    radians = np.deg2rad(modeled_angle)
    vectors = np.column_stack([np.sin(radians), np.cos(radians)])
    return vectors, modeled_angle, yaw_heading


def modeled_to_absolute_angle(
    modeled_angle_deg: np.ndarray,
    target_mode: str,
    yaw_heading_deg: np.ndarray | None,
) -> np.ndarray:
    """Convert a modeled direction back to the absolute Wind_angle convention."""
    modeled_angle_deg = np.asarray(modeled_angle_deg, dtype=float) % 360.0
    if target_mode == "absolute":
        return modeled_angle_deg
    if target_mode == "relative_yaw":
        if yaw_heading_deg is None:
            raise ValueError("relative_yaw predictions require yaw headings for reconstruction.")
        return (modeled_angle_deg + np.asarray(yaw_heading_deg, dtype=float)) % 360.0
    raise ValueError(f"Unknown direction target mode: {target_mode}")


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return float(
        (np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) + 360.0)
        % 360.0
    )


def json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_metrics(metrics: Dict[str, Any], output_dir: Path) -> None:
    clean = {key: json_value(value) for key, value in metrics.items()}
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(clean, file, indent=2)
    pd.DataFrame([clean]).to_csv(output_dir / "model_metrics.csv", index=False)


def make_group_id(df: pd.DataFrame) -> pd.Series:
    source = (
        df["Source_dataset"].astype("string").fillna("unknown")
        if "Source_dataset" in df.columns
        else pd.Series("unknown", index=df.index, dtype="string")
    )
    flight = df["Flight_ID"].astype("string").fillna("missing")
    return source + "::" + flight


def load_standardized_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = set(BASE_FEATURES + TARGET_COLUMNS + ["Flight_ID"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "Source_dataset" not in df.columns:
        df["Source_dataset"] = "unknown"
    for col in BASE_FEATURES + TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Wind_speed", "Wind_angle", "Flight_ID"]).copy()
    df = df[df["Wind_speed"] >= 0.0].copy()
    df["Wind_angle"] = df["Wind_angle"] % 360.0
    if df.empty:
        raise ValueError("No usable labeled rows remain after data validation.")
    return df


def add_engineered_features(
    df: pd.DataFrame, attitude_angle_unit: str = "auto"
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Create causal features using only the current and preceding rows of each flight."""
    data = df.copy()
    data["_original_row"] = np.arange(len(data), dtype=np.int64)
    data["_Group_ID"] = make_group_id(data)

    data["_elapsed_numeric"] = (
        pd.to_numeric(data["Elapsed_s"], errors="coerce")
        if "Elapsed_s" in data.columns
        else np.nan
    )
    data["_timestamp_parsed"] = (
        pd.to_datetime(data["Timestamp"], errors="coerce")
        if "Timestamp" in data.columns
        else pd.NaT
    )
    data = data.sort_values(
        ["_Group_ID", "_elapsed_numeric", "_timestamp_parsed", "_original_row"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    group = data.groupby("_Group_ID", sort=False, observed=True)
    data["Sample_index"] = group.cumcount().astype(float)
    data["Sample_index_log1p"] = np.log1p(data["Sample_index"])

    x0 = group["X"].transform("first")
    y0 = group["Y"].transform("first")
    z0 = group["Z"].transform("first")
    if "Coordinate_frame" in data.columns:
        frame_label = data["Coordinate_frame"].astype("string").str.strip().str.lower()
        labeled_geo = frame_label.str.contains(
            "geographic|wgs|lat|lon|gps", na=False, regex=True
        )
        # Any other non-empty label is an explicit declaration of a non-geographic
        # frame (e.g. cartesian_meters, local ENU) and must veto the inference below.
        labeled_non_geo = frame_label.notna() & frame_label.ne("") & ~labeled_geo
    else:
        labeled_geo = pd.Series(False, index=data.index)
        labeled_non_geo = pd.Series(False, index=data.index)
    x_median = group["X"].transform("median").abs()
    y_median = group["Y"].transform("median").abs()
    # A real lat/lon flight spans far less than one degree, so require a tiny
    # per-flight coordinate range in addition to the magnitude checks.
    x_range = group["X"].transform("max") - group["X"].transform("min")
    y_range = group["Y"].transform("max") - group["Y"].transform("min")
    inferred_geo = (
        data["X"].between(-180.0, 180.0)
        & data["Y"].between(-90.0, 90.0)
        & (x_median > 20.0)
        & (y_median > 10.0)
        & (x_range < 0.5)
        & (y_range < 0.5)
    )
    is_geo = labeled_geo | (inferred_geo & ~labeled_non_geo)
    mean_lat_rad = np.deg2rad((data["Y"] + y0) / 2.0)
    east_geo = EARTH_RADIUS_M * np.deg2rad(data["X"] - x0) * np.cos(mean_lat_rad)
    north_geo = EARTH_RADIUS_M * np.deg2rad(data["Y"] - y0)
    data["Local_east_m"] = np.where(is_geo, east_geo, data["X"] - x0)
    data["Local_north_m"] = np.where(is_geo, north_geo, data["Y"] - y0)
    data["Relative_altitude_m"] = data["Z"] - z0
    data["Coordinate_is_geographic"] = is_geo.astype(float)
    data["Position_valid"] = data[["X", "Y", "Z"]].notna().all(axis=1).astype(float)

    if attitude_angle_unit not in {"auto", "radians", "degrees"}:
        raise ValueError("attitude_angle_unit must be auto, radians, or degrees.")
    resolved_angle_unit = attitude_angle_unit
    if attitude_angle_unit == "auto":
        finite_yaw = data["Yaw"].to_numpy(dtype=float)
        finite_yaw = finite_yaw[np.isfinite(finite_yaw)]
        yaw_q99 = float(np.quantile(np.abs(finite_yaw), 0.99)) if len(finite_yaw) else 0.0
        threshold = 2.0 * np.pi * 1.25
        resolved_angle_unit = "degrees" if yaw_q99 > threshold else "radians"
        if np.pi < yaw_q99 <= 6.0 * threshold:
            # Unwrapped/accumulated radian yaw can cross the degree threshold and
            # bounded degree yaw can sit just above it; auto detection is unsafe here.
            print(
                f"WARNING: auto attitude-unit detection is ambiguous (|yaw| q99 = {yaw_q99:.2f}; "
                f"resolved to {resolved_angle_unit}). Pass --attitude-angle-unit explicitly "
                "after confirming the dataset's units."
            )
    yaw_radians = np.deg2rad(data["Yaw"]) if resolved_angle_unit == "degrees" else data["Yaw"]
    roll_radians = np.deg2rad(data["Roll"]) if resolved_angle_unit == "degrees" else data["Roll"]
    pitch_radians = np.deg2rad(data["Pitch"]) if resolved_angle_unit == "degrees" else data["Pitch"]
    data["Yaw_sin"] = np.sin(yaw_radians)
    data["Yaw_cos"] = np.cos(yaw_radians)
    data["Tilt_magnitude"] = np.sqrt(data["Roll"] ** 2 + data["Pitch"] ** 2)
    data["Battery_power_W"] = data["Battery_V"] * data["Battery_C"]
    data["Battery_current_abs"] = data["Battery_C"].abs()

    elapsed_dt = group["_elapsed_numeric"].diff()
    timestamp_dt = group["_timestamp_parsed"].diff().dt.total_seconds()
    data["Delta_time_s"] = elapsed_dt.where(
        elapsed_dt > 0.0, timestamp_dt.where(timestamp_dt > 0.0)
    )
    usable_time_fraction = float(data["Delta_time_s"].notna().mean())

    yaw_series = pd.Series(np.asarray(yaw_radians, dtype=float), index=data.index)
    roll_series = pd.Series(np.asarray(roll_radians, dtype=float), index=data.index)
    pitch_series = pd.Series(np.asarray(pitch_radians, dtype=float), index=data.index)
    previous_yaw = yaw_series.groupby(data["_Group_ID"], sort=False).shift(1)
    data["Yaw_delta_rad"] = np.arctan2(
        np.sin(yaw_series - previous_yaw), np.cos(yaw_series - previous_yaw)
    )
    data["Roll_delta_rad"] = roll_series.groupby(data["_Group_ID"], sort=False).diff()
    data["Pitch_delta_rad"] = pitch_series.groupby(data["_Group_ID"], sort=False).diff()

    dynamic_columns = [
        "Local_east_m", "Local_north_m", "Relative_altitude_m", "Roll", "Pitch",
        "Yaw_sin", "Yaw_cos", "Tilt_magnitude", "Battery_V", "Battery_C",
        "Battery_power_W",
    ]
    for col in dynamic_columns:
        data[f"d1_{col}"] = group[col].diff()
        data[f"d2_{col}"] = data.groupby(
            "_Group_ID", sort=False, observed=True
        )[f"d1_{col}"].diff()

    time_feature_columns: List[str] = []
    if usable_time_fraction > 0.01:
        dt = data["Delta_time_s"]
        for col in ["Local_east_m", "Local_north_m", "Relative_altitude_m"]:
            velocity_name = f"velocity_{col}_per_s"
            acceleration_name = f"acceleration_{col}_per_s2"
            data[velocity_name] = data[f"d1_{col}"] / dt
            data[acceleration_name] = data.groupby(
                "_Group_ID", sort=False, observed=True
            )[velocity_name].diff() / dt
            time_feature_columns.extend([velocity_name, acceleration_name])
        data["Horizontal_speed_m_per_s"] = np.sqrt(
            data["velocity_Local_east_m_per_s"] ** 2
            + data["velocity_Local_north_m_per_s"] ** 2
        )
        course_rad = np.arctan2(
            data["velocity_Local_east_m_per_s"], data["velocity_Local_north_m_per_s"]
        )
        data["Course_sin"] = np.sin(course_rad)
        data["Course_cos"] = np.cos(course_rad)
        data["Yaw_rate_rad_per_s"] = data["Yaw_delta_rad"] / dt
        data["Roll_rate_rad_per_s"] = data["Roll_delta_rad"] / dt
        data["Pitch_rate_rad_per_s"] = data["Pitch_delta_rad"] / dt
        data["Delta_time_log1p"] = np.log1p(data["Delta_time_s"])
        time_feature_columns.extend(
            [
                "Horizontal_speed_m_per_s", "Course_sin", "Course_cos",
                "Yaw_rate_rad_per_s", "Roll_rate_rad_per_s", "Pitch_rate_rad_per_s",
                "Delta_time_s", "Delta_time_log1p",
            ]
        )

    rolling_columns = [
        "Roll", "Pitch", "Tilt_magnitude", "Relative_altitude_m", "Battery_V",
        "Battery_C", "Battery_power_W",
    ]
    if time_feature_columns:
        rolling_columns.extend(
            [
                "Horizontal_speed_m_per_s", "Yaw_rate_rad_per_s",
                "Roll_rate_rad_per_s", "Pitch_rate_rad_per_s",
            ]
        )
    rolling_feature_columns: List[str] = []
    rolling_values: Dict[str, pd.Series] = {}
    for col in rolling_columns:
        for window in (5, 15, 30):
            grouped_col = data.groupby("_Group_ID", sort=False, observed=True)[col]
            mean_name = f"{col}_mean_{window}"
            std_name = f"{col}_std_{window}"
            rolling_values[mean_name] = grouped_col.rolling(window=window, min_periods=1).mean().reset_index(
                level=0, drop=True
            )
            rolling_values[std_name] = grouped_col.rolling(window=window, min_periods=2).std().reset_index(
                level=0, drop=True
            )
            rolling_feature_columns.extend([mean_name, std_name])
    if rolling_values:
        data = pd.concat([data, pd.DataFrame(rolling_values, index=data.index)], axis=1)

    feature_columns = [
        "Z", "Roll", "Pitch", "Yaw_sin", "Yaw_cos", "Tilt_magnitude", "Battery_V",
        "Battery_C", "Battery_power_W", "Battery_current_abs", "Local_east_m",
        "Local_north_m", "Relative_altitude_m", "Coordinate_is_geographic",
        "Position_valid", "Sample_index", "Sample_index_log1p",
        "Yaw_delta_rad", "Roll_delta_rad", "Pitch_delta_rad",
    ]
    feature_columns.extend([f"d1_{col}" for col in dynamic_columns])
    feature_columns.extend([f"d2_{col}" for col in dynamic_columns])
    feature_columns.extend(time_feature_columns)
    feature_columns.extend(rolling_feature_columns)

    # Any divisions by unusually small time deltas are treated as missing and imputed later.
    data[feature_columns] = data[feature_columns].replace([np.inf, -np.inf], np.nan)
    metadata = {
        "usable_time_fraction": usable_time_fraction,
        "time_features_enabled": bool(time_feature_columns),
        "feature_count": len(feature_columns),
        "attitude_angle_unit_resolved": resolved_angle_unit,
    }
    return data, feature_columns, metadata


def load_split(engineered: pd.DataFrame, manifest_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    required = {"Group_ID", "split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Split manifest is missing columns: {missing}")
    if manifest["Group_ID"].duplicated().any():
        raise ValueError("Split manifest contains duplicate Group_ID values.")
    split_map = manifest.set_index("Group_ID")["split"].astype(str).str.lower()
    unknown = sorted(set(split_map) - {"train", "test"})
    if unknown:
        raise ValueError(f"Unknown split labels: {unknown}")
    row_split = engineered["_Group_ID"].map(split_map)
    missing_groups = sorted(engineered.loc[row_split.isna(), "_Group_ID"].unique())
    if missing_groups:
        raise ValueError(
            "Split manifest does not cover every flight. Missing examples: "
            + ", ".join(missing_groups[:5])
        )
    train_idx = np.flatnonzero(row_split.to_numpy() == "train")
    test_idx = np.flatnonzero(row_split.to_numpy() == "test")
    if not len(train_idx) or not len(test_idx):
        raise ValueError("Split manifest must include at least one train and one test flight.")
    train_groups = set(engineered.iloc[train_idx]["_Group_ID"])
    test_groups = set(engineered.iloc[test_idx]["_Group_ID"])
    if train_groups & test_groups:
        raise RuntimeError("Flight leakage detected between train and test sets.")
    return train_idx, test_idx, manifest


def equal_flight_weights(groups: Iterable[str]) -> np.ndarray:
    groups = pd.Series(list(groups), dtype="string")
    counts = groups.map(groups.value_counts()).to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / np.mean(weights)


def direction_metrics_for_mask(
    actual_angle: np.ndarray,
    predicted_angle: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> Dict[str, Any]:
    n = int(np.sum(mask))
    if n == 0:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_circular_mae_deg": None,
            f"{prefix}_circular_rmse_deg": None,
            f"{prefix}_within_15deg_fraction": None,
            f"{prefix}_within_30deg_fraction": None,
        }
    error = circular_difference_deg(actual_angle[mask], predicted_angle[mask])
    abs_error = np.abs(error)
    return {
        f"{prefix}_rows": n,
        f"{prefix}_circular_mae_deg": float(np.mean(abs_error)),
        f"{prefix}_circular_rmse_deg": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_within_15deg_fraction": float(np.mean(abs_error <= 15.0)),
        f"{prefix}_within_30deg_fraction": float(np.mean(abs_error <= 30.0)),
    }


def evaluate_predictions(
    actual_speed: np.ndarray,
    predicted_speed: np.ndarray,
    actual_angle: np.ndarray,
    predicted_angle: np.ndarray,
) -> Dict[str, Any]:
    actual_speed = np.asarray(actual_speed, dtype=float)
    predicted_speed = np.asarray(predicted_speed, dtype=float)
    actual_angle = np.asarray(actual_angle, dtype=float)
    predicted_angle = np.asarray(predicted_angle, dtype=float)
    if len(actual_speed) == 0:
        raise ValueError("Cannot evaluate an empty prediction set.")
    metrics: Dict[str, Any] = {
        "speed_mae_mps": float(mean_absolute_error(actual_speed, predicted_speed)),
        "speed_rmse_mps": float(np.sqrt(mean_squared_error(actual_speed, predicted_speed))),
        "speed_r2": float(r2_score(actual_speed, predicted_speed)) if len(actual_speed) > 1 else None,
    }
    metrics.update(
        direction_metrics_for_mask(
            actual_angle, predicted_angle, np.ones(len(actual_speed), dtype=bool), "direction_all"
        )
    )
    for threshold in (0.5, 1.0, 2.0):
        safe_name = str(threshold).replace(".", "p")
        metrics.update(
            direction_metrics_for_mask(
                actual_angle,
                predicted_angle,
                actual_speed >= threshold,
                f"direction_speed_ge_{safe_name}_mps",
            )
        )
    return metrics


def calculate_per_flight_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for group_id, part in predictions.groupby("_Group_ID", sort=True, observed=True):
        speed = part["Wind_speed"].to_numpy(dtype=float)
        pred_speed = part["Predicted_wind_speed"].to_numpy(dtype=float)
        angle = part["Wind_angle"].to_numpy(dtype=float)
        pred_angle = part["Predicted_wind_angle"].to_numpy(dtype=float)
        abs_angle_error = np.abs(circular_difference_deg(angle, pred_angle))
        row: Dict[str, Any] = {
            "Group_ID": group_id,
            "Source_dataset": part["Source_dataset"].iloc[0],
            "Flight_ID": part["Flight_ID"].iloc[0],
            "rows": len(part),
            "speed_mae_mps": float(mean_absolute_error(speed, pred_speed)),
            "speed_rmse_mps": float(np.sqrt(mean_squared_error(speed, pred_speed))),
            "direction_circular_mae_deg": float(np.mean(abs_angle_error)),
            "direction_within_30deg_fraction": float(np.mean(abs_angle_error <= 30.0)),
        }
        for threshold in (0.5, 1.0, 2.0):
            safe_name = str(threshold).replace(".", "p")
            mask = speed >= threshold
            row[f"direction_speed_ge_{safe_name}_mps_rows"] = int(mask.sum())
            row[f"direction_speed_ge_{safe_name}_mps_circular_mae_deg"] = (
                float(np.mean(abs_angle_error[mask])) if mask.any() else np.nan
            )
            row[f"direction_speed_ge_{safe_name}_mps_within_30deg_fraction"] = (
                float(np.mean(abs_angle_error[mask] <= 30.0)) if mask.any() else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_frame(
    engineered: pd.DataFrame,
    endpoint_indices: np.ndarray,
    predicted_speed: np.ndarray,
    predicted_angle: np.ndarray,
    comparison_sequence_length: int,
) -> pd.DataFrame:
    keep = [
        col for col in [
            "_Group_ID", "Source_dataset", "Flight_ID", "Timestamp", "Elapsed_s", "X", "Y", "Z",
            "Sample_index", "Wind_speed", "Wind_angle",
        ] if col in engineered.columns
    ]
    out = engineered.iloc[endpoint_indices][keep].copy()
    out["Predicted_wind_speed"] = np.asarray(predicted_speed, dtype=float)
    out["Predicted_wind_angle"] = np.asarray(predicted_angle, dtype=float)
    out["Speed_error_mps"] = out["Predicted_wind_speed"] - out["Wind_speed"]
    out["Angle_error_deg"] = circular_difference_deg(
        out["Wind_angle"].to_numpy(), out["Predicted_wind_angle"].to_numpy()
    )
    out["Common_sequence_eligible"] = (
        out["Sample_index"].to_numpy(dtype=float) >= comparison_sequence_length - 1
    )
    return out


def append_prediction_metrics(
    metrics: Dict[str, Any], predictions: pd.DataFrame, *, include_common_scope: bool = True
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    overall = evaluate_predictions(
        predictions["Wind_speed"].to_numpy(),
        predictions["Predicted_wind_speed"].to_numpy(),
        predictions["Wind_angle"].to_numpy(),
        predictions["Predicted_wind_angle"].to_numpy(),
    )
    metrics.update(overall)
    per_flight = calculate_per_flight_metrics(predictions)
    metrics["flight_balanced_speed_mae_mps"] = float(per_flight["speed_mae_mps"].mean())
    metrics["flight_balanced_direction_circular_mae_deg"] = float(
        per_flight["direction_circular_mae_deg"].mean()
    )
    for threshold in (0.5, 1.0, 2.0):
        safe_name = str(threshold).replace(".", "p")
        mae_col = f"direction_speed_ge_{safe_name}_mps_circular_mae_deg"
        within_col = f"direction_speed_ge_{safe_name}_mps_within_30deg_fraction"
        metrics[f"flight_balanced_{mae_col}"] = float(per_flight[mae_col].mean())
        metrics[f"flight_balanced_{within_col}"] = float(per_flight[within_col].mean())

    if "Predicted_direction_confidence" in predictions.columns:
        confidence = predictions["Predicted_direction_confidence"].to_numpy(dtype=float)
        metrics["predicted_direction_confidence_mean"] = float(np.mean(confidence))
        metrics["predicted_direction_confidence_median"] = float(np.median(confidence))
        metrics["predicted_direction_confidence_below_0p2_fraction"] = float(
            np.mean(confidence < 0.2)
        )

    if include_common_scope:
        common = predictions[predictions["Common_sequence_eligible"]].copy()
        metrics["common_rows"] = len(common)
        if len(common):
            common_metrics = evaluate_predictions(
                common["Wind_speed"].to_numpy(),
                common["Predicted_wind_speed"].to_numpy(),
                common["Wind_angle"].to_numpy(),
                common["Predicted_wind_angle"].to_numpy(),
            )
            metrics.update({f"common_{k}": v for k, v in common_metrics.items()})
            common_pf = calculate_per_flight_metrics(common)
            metrics["common_flight_balanced_speed_mae_mps"] = float(
                common_pf["speed_mae_mps"].mean()
            )
            metrics["common_flight_balanced_direction_circular_mae_deg"] = float(
                common_pf["direction_circular_mae_deg"].mean()
            )
            for threshold in (0.5, 1.0, 2.0):
                safe_name = str(threshold).replace(".", "p")
                mae_col = f"direction_speed_ge_{safe_name}_mps_circular_mae_deg"
                within_col = f"direction_speed_ge_{safe_name}_mps_within_30deg_fraction"
                metrics[f"common_flight_balanced_{mae_col}"] = float(
                    common_pf[mae_col].mean()
                )
                metrics[f"common_flight_balanced_{within_col}"] = float(
                    common_pf[within_col].mean()
                )
    return metrics, per_flight
