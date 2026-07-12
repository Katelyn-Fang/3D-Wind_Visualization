"""
Train wind-speed and wind-direction models from standardized drone telemetry.

Version 2.1.0

Design goals
------------
1. Keep entire flights together during train/test splitting.
2. Make the split and model reproducible through --random-seed.
3. Allow either --train-size or the backward-compatible --test-size.
4. Predict circular wind direction through sine/cosine targets.
5. Use only current and preceding telemetry rows for engineered features.
6. Report model accuracy, simple baselines, and per-flight results.

Primary outputs
---------------
<output-dir>/
    wind_model.joblib
    metrics.json
    test_predictions.csv
    per_flight_metrics.csv
    flight_split_manifest.csv
    feature_importance_speed.csv       (optional)
    feature_importance_direction.csv   (optional)
    wind_model_all_data.joblib         (optional, with --refit-all)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

SCRIPT_VERSION = "2.1.0"
EARTH_RADIUS_M = 6_371_000.0

BASE_FEATURES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Battery_V", "Battery_C"]
TARGET_COLUMNS = ["Wind_speed", "Wind_angle"]


def circular_difference_deg(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Signed shortest angular difference, predicted minus actual, in [-180, 180)."""
    return (predicted - actual + 180.0) % 360.0 - 180.0


def vectors_to_angle_deg(vectors: np.ndarray) -> np.ndarray:
    """Convert columns [sin(angle), cos(angle)] to angles in [0, 360)."""
    vectors = np.asarray(vectors, dtype=float)
    return (np.rad2deg(np.arctan2(vectors[:, 0], vectors[:, 1])) + 360.0) % 360.0


def direction_neg_circular_mae_scorer(estimator: Any, X: pd.DataFrame, y: np.ndarray) -> float:
    """Permutation-importance scorer; larger values are better."""
    actual_angle = vectors_to_angle_deg(y)
    predicted_angle = vectors_to_angle_deg(estimator.predict(X))
    return -float(np.mean(np.abs(circular_difference_deg(actual_angle, predicted_angle))))


def json_value(value: Any) -> Any:
    """Convert NumPy/Pandas scalars and non-finite values into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def resolve_split_sizes(train_size: Optional[float], test_size: Optional[float]) -> Tuple[float, float]:
    """Resolve train/test fractions while supporting either command-line convention."""
    if train_size is None and test_size is None:
        train_size = 0.80
        test_size = 0.20
    elif train_size is None:
        test_size = float(test_size)
        train_size = 1.0 - test_size
    elif test_size is None:
        train_size = float(train_size)
        test_size = 1.0 - train_size
    else:
        train_size = float(train_size)
        test_size = float(test_size)
        if not np.isclose(train_size + test_size, 1.0, atol=1e-9):
            raise ValueError("When both are supplied, --train-size and --test-size must sum to 1.0.")

    if not 0.0 < train_size < 1.0:
        raise ValueError("--train-size must be strictly between 0 and 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be strictly between 0 and 1.")
    return float(train_size), float(test_size)


def make_group_id(df: pd.DataFrame) -> pd.Series:
    """Create collision-resistant flight groups across one or more source datasets."""
    source = (
        df["Source_dataset"].astype("string").fillna("unknown")
        if "Source_dataset" in df.columns
        else pd.Series("unknown", index=df.index, dtype="string")
    )
    flight = df["Flight_ID"].astype("string").fillna("missing")
    return source + "::" + flight


def add_engineered_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Create causal features from current and prior rows within each flight."""
    data = df.copy()
    data["_original_row"] = np.arange(len(data), dtype=np.int64)
    data["_Group_ID"] = make_group_id(data)

    if "Elapsed_s" in data.columns:
        data["_elapsed_numeric"] = pd.to_numeric(data["Elapsed_s"], errors="coerce")
    else:
        data["_elapsed_numeric"] = np.nan

    if "Timestamp" in data.columns:
        data["_timestamp_parsed"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    else:
        data["_timestamp_parsed"] = pd.NaT

    # Stable sorting preserves original order when usable time is unavailable.
    data = data.sort_values(
        ["_Group_ID", "_elapsed_numeric", "_timestamp_parsed", "_original_row"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    group = data.groupby("_Group_ID", sort=False, observed=True)
    data["Sample_index"] = group.cumcount().astype(float)
    data["Sample_index_log1p"] = np.log1p(data["Sample_index"])

    # Local coordinates relative to the first valid sample of each flight.
    x0 = group["X"].transform("first")
    y0 = group["Y"].transform("first")
    z0 = group["Z"].transform("first")

    if "Coordinate_frame" in data.columns:
        labeled_geo = data["Coordinate_frame"].astype("string").str.contains(
            "geographic", case=False, na=False
        )
    else:
        labeled_geo = pd.Series(False, index=data.index)

    x_median = group["X"].transform("median").abs()
    y_median = group["Y"].transform("median").abs()
    inferred_geo = (
        data["X"].between(-180.0, 180.0)
        & data["Y"].between(-90.0, 90.0)
        & (x_median > 20.0)
        & (y_median > 10.0)
    )
    is_geo = labeled_geo | inferred_geo

    mean_lat_rad = np.deg2rad((data["Y"] + y0) / 2.0)
    east_geo = EARTH_RADIUS_M * np.deg2rad(data["X"] - x0) * np.cos(mean_lat_rad)
    north_geo = EARTH_RADIUS_M * np.deg2rad(data["Y"] - y0)
    east_cart = data["X"] - x0
    north_cart = data["Y"] - y0

    data["Local_east_m"] = np.where(is_geo, east_geo, east_cart)
    data["Local_north_m"] = np.where(is_geo, north_geo, north_cart)
    data["Relative_altitude_m"] = data["Z"] - z0
    data["Coordinate_is_geographic"] = is_geo.astype(float)
    data["Position_valid"] = data[["X", "Y", "Z"]].notna().all(axis=1).astype(float)

    # Circular attitude encoding and physically meaningful simple interactions.
    data["Yaw_sin"] = np.sin(data["Yaw"])
    data["Yaw_cos"] = np.cos(data["Yaw"])
    data["Tilt_magnitude"] = np.sqrt(data["Roll"] ** 2 + data["Pitch"] ** 2)
    data["Battery_power_W"] = data["Battery_V"] * data["Battery_C"]
    data["Battery_current_abs"] = data["Battery_C"].abs()

    # Prefer elapsed time; fall back to timestamp differences when positive.
    elapsed_dt = group["_elapsed_numeric"].diff()
    timestamp_dt = group["_timestamp_parsed"].diff().dt.total_seconds()
    data["Delta_time_s"] = elapsed_dt.where(elapsed_dt > 0.0, timestamp_dt.where(timestamp_dt > 0.0))
    usable_time_fraction = float(data["Delta_time_s"].notna().mean())

    dynamic_columns = [
        "Local_east_m",
        "Local_north_m",
        "Relative_altitude_m",
        "Roll",
        "Pitch",
        "Yaw_sin",
        "Yaw_cos",
        "Tilt_magnitude",
        "Battery_V",
        "Battery_C",
        "Battery_power_W",
    ]

    # Row-to-row changes are not called velocity unless a positive time interval exists.
    for col in dynamic_columns:
        data[f"d1_{col}"] = group[col].diff()
        data[f"d2_{col}"] = data.groupby("_Group_ID", sort=False, observed=True)[f"d1_{col}"].diff()

    time_feature_columns: List[str] = []
    if usable_time_fraction > 0.01:
        dt = data["Delta_time_s"]
        for col in ["Local_east_m", "Local_north_m", "Relative_altitude_m"]:
            velocity_name = f"velocity_{col}_per_s"
            acceleration_name = f"acceleration_{col}_per_s2"
            data[velocity_name] = data[f"d1_{col}"] / dt
            data[acceleration_name] = (
                data.groupby("_Group_ID", sort=False, observed=True)[velocity_name].diff() / dt
            )
            time_feature_columns.extend([velocity_name, acceleration_name])
        data["Delta_time_log1p"] = np.log1p(data["Delta_time_s"])
        time_feature_columns.extend(["Delta_time_s", "Delta_time_log1p"])

    # Trailing windows include the current observation and never use future rows.
    rolling_columns = [
        "Roll",
        "Pitch",
        "Tilt_magnitude",
        "Relative_altitude_m",
        "Battery_V",
        "Battery_C",
        "Battery_power_W",
    ]
    rolling_feature_columns: List[str] = []
    for col in rolling_columns:
        for window in (5, 15):
            grouped_col = data.groupby("_Group_ID", sort=False, observed=True)[col]
            mean_name = f"{col}_mean_{window}"
            std_name = f"{col}_std_{window}"
            data[mean_name] = (
                grouped_col.rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            data[std_name] = (
                grouped_col.rolling(window=window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
            )
            rolling_feature_columns.extend([mean_name, std_name])

    feature_columns = [
        "Z",
        "Roll",
        "Pitch",
        "Yaw_sin",
        "Yaw_cos",
        "Tilt_magnitude",
        "Battery_V",
        "Battery_C",
        "Battery_power_W",
        "Battery_current_abs",
        "Local_east_m",
        "Local_north_m",
        "Relative_altitude_m",
        "Coordinate_is_geographic",
        "Position_valid",
        "Sample_index",
        "Sample_index_log1p",
    ]
    feature_columns.extend([f"d1_{col}" for col in dynamic_columns])
    feature_columns.extend([f"d2_{col}" for col in dynamic_columns])
    feature_columns.extend(time_feature_columns)
    feature_columns.extend(rolling_feature_columns)

    metadata = {
        "usable_time_fraction": usable_time_fraction,
        "time_features_enabled": bool(time_feature_columns),
        "feature_count": len(feature_columns),
    }
    return data, feature_columns, metadata


def make_regressor(
    n_estimators: int,
    min_samples_leaf: int,
    max_depth: Optional[int],
    max_features: float,
    n_jobs: int,
    random_seed: int,
) -> Pipeline:
    """Build the tree ensemble and missing-value imputation pipeline."""
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_seed,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", model),
    ])


def equal_flight_weights(groups: pd.Series) -> np.ndarray:
    """Give each flight equal total influence regardless of its row count."""
    counts = groups.map(groups.value_counts()).to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / np.mean(weights)


def direction_metrics_for_mask(
    actual_angle: np.ndarray,
    predicted_angle: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> Dict[str, Any]:
    """Calculate direction metrics for a selected wind-speed range."""
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
        f"{prefix}_circular_rmse_deg": float(np.sqrt(np.mean(error ** 2))),
        f"{prefix}_within_15deg_fraction": float(np.mean(abs_error <= 15.0)),
        f"{prefix}_within_30deg_fraction": float(np.mean(abs_error <= 30.0)),
    }


def evaluate_predictions(
    actual_speed: np.ndarray,
    predicted_speed: np.ndarray,
    actual_angle: np.ndarray,
    predicted_angle: np.ndarray,
) -> Dict[str, Any]:
    """Return row-weighted metrics, including direction accuracy by wind strength."""
    metrics: Dict[str, Any] = {
        "speed_mae_mps": float(mean_absolute_error(actual_speed, predicted_speed)),
        "speed_rmse_mps": float(np.sqrt(mean_squared_error(actual_speed, predicted_speed))),
        "speed_r2": float(r2_score(actual_speed, predicted_speed)),
    }
    all_rows = np.ones(len(actual_speed), dtype=bool)
    metrics.update(direction_metrics_for_mask(actual_angle, predicted_angle, all_rows, "direction_all"))
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
    """Calculate accuracy separately for each held-out flight."""
    rows: List[Dict[str, Any]] = []
    for group_id, part in predictions.groupby("_Group_ID", sort=True, observed=True):
        speed = part["Wind_speed"].to_numpy(dtype=float)
        pred_speed = part["Predicted_wind_speed"].to_numpy(dtype=float)
        angle = part["Wind_angle"].to_numpy(dtype=float)
        pred_angle = part["Predicted_wind_angle"].to_numpy(dtype=float)
        angle_error = np.abs(circular_difference_deg(angle, pred_angle))
        rows.append({
            "Group_ID": group_id,
            "Source_dataset": part["Source_dataset"].iloc[0],
            "Flight_ID": part["Flight_ID"].iloc[0],
            "rows": len(part),
            "speed_mae_mps": mean_absolute_error(speed, pred_speed),
            "speed_rmse_mps": np.sqrt(mean_squared_error(speed, pred_speed)),
            "direction_circular_mae_deg": float(np.mean(angle_error)),
            "direction_within_30deg_fraction": float(np.mean(angle_error <= 30.0)),
        })
    return pd.DataFrame(rows)


def circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Circular mean angle in degrees."""
    radians = np.deg2rad(angles_deg)
    return float((np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) + 360.0) % 360.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train reproducible flight-grouped wind models from standardized drone telemetry."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--data", default="DJI_primary_standardized.csv")
    parser.add_argument("--output-dir", default="model_output")
    parser.add_argument("--train-size", type=float, default=None, help="Fraction of flights used for training; default 0.80.")
    parser.add_argument("--test-size", type=float, default=None, help="Backward-compatible test fraction; default 0.20.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=20, help="Tree depth; use 0 for unlimited depth.")
    parser.add_argument("--max-features", type=float, default=0.80)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--direction-min-speed", type=float, default=0.50, help="Down-weight direction labels below this wind speed.")
    parser.add_argument("--permutation-importance", action="store_true")
    parser.add_argument("--importance-sample-size", type=int, default=5000)
    parser.add_argument("--refit-all", action="store_true", help="After evaluation, also fit and save a model using all labeled flights.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print(f"Running script: {Path(__file__).resolve()}")
    print(f"Script version: {SCRIPT_VERSION}")

    try:
        train_size, test_size = resolve_split_sizes(args.train_size, args.test_size)
    except ValueError as exc:
        parser.error(str(exc))

    if args.n_estimators < 1:
        parser.error("--n-estimators must be at least 1.")
    if args.min_samples_leaf < 1:
        parser.error("--min-samples-leaf must be at least 1.")
    if not 0.0 < args.max_features <= 1.0:
        parser.error("--max-features must be in (0, 1].")
    if args.direction_min_speed < 0.0:
        parser.error("--direction-min-speed cannot be negative.")

    max_depth = None if args.max_depth == 0 else args.max_depth
    if max_depth is not None and max_depth < 1:
        parser.error("--max-depth must be 0 (unlimited) or a positive integer.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path.resolve()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
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

    engineered, feature_columns, feature_metadata = add_engineered_features(df)
    groups = engineered["_Group_ID"]
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("At least two distinct flights are required for a train/test split.")

    X = engineered[feature_columns]
    speed_target = engineered["Wind_speed"].to_numpy(dtype=float)
    angle_deg = engineered["Wind_angle"].to_numpy(dtype=float)
    angle_rad = np.deg2rad(angle_deg)
    direction_target = np.column_stack([np.sin(angle_rad), np.cos(angle_rad)])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=args.random_seed)
    train_idx, test_idx = next(splitter.split(X, groups=groups))

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Flight leakage detected between train and test sets: {sorted(overlap)[:5]}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    speed_train = speed_target[train_idx]
    speed_test = speed_target[test_idx]
    direction_train = direction_target[train_idx]
    angle_test = angle_deg[test_idx]

    flight_weights = equal_flight_weights(groups.iloc[train_idx])
    if args.direction_min_speed > 0.0:
        direction_reliability = np.clip(speed_train / args.direction_min_speed, 0.05, 1.0)
    else:
        direction_reliability = np.ones_like(speed_train)
    direction_weights = flight_weights * direction_reliability
    direction_weights = direction_weights / np.mean(direction_weights)

    speed_model = make_regressor(
        args.n_estimators, args.min_samples_leaf, max_depth,
        args.max_features, args.n_jobs, args.random_seed,
    )
    direction_model = make_regressor(
        args.n_estimators, args.min_samples_leaf, max_depth,
        args.max_features, args.n_jobs, args.random_seed,
    )

    print(f"Rows: {len(engineered):,} total; {len(train_idx):,} train; {len(test_idx):,} test")
    print(f"Flights: {unique_groups} total; {len(train_groups)} train; {len(test_groups)} test")
    print(f"Resolved split: train={train_size:.3f}, test={test_size:.3f}; random seed={args.random_seed}")
    if not feature_metadata["time_features_enabled"]:
        print("Timing note: no usable row-level time intervals were found; physical velocity/acceleration features were not created.")

    speed_model.fit(X_train, speed_train, model__sample_weight=flight_weights)
    direction_model.fit(X_train, direction_train, model__sample_weight=direction_weights)

    predicted_speed = np.maximum(speed_model.predict(X_test), 0.0)
    predicted_direction_vectors = direction_model.predict(X_test)
    predicted_angle = vectors_to_angle_deg(predicted_direction_vectors)

    metrics = evaluate_predictions(speed_test, predicted_speed, angle_test, predicted_angle)

    # Honest simple baselines derived only from the training set.
    baseline_speed = float(np.average(speed_train, weights=flight_weights))
    baseline_angle = circular_mean_deg(angle_deg[train_idx])
    baseline_speed_predictions = np.full_like(speed_test, baseline_speed, dtype=float)
    baseline_angle_predictions = np.full_like(angle_test, baseline_angle, dtype=float)
    baseline_metrics = evaluate_predictions(
        speed_test, baseline_speed_predictions, angle_test, baseline_angle_predictions
    )
    metrics["baseline_speed_constant_mps"] = baseline_speed
    metrics["baseline_direction_constant_deg"] = baseline_angle
    metrics["baseline_speed_mae_mps"] = baseline_metrics["speed_mae_mps"]
    metrics["baseline_direction_all_circular_mae_deg"] = baseline_metrics["direction_all_circular_mae_deg"]
    metrics["speed_mae_improvement_vs_baseline_fraction"] = (
        (baseline_metrics["speed_mae_mps"] - metrics["speed_mae_mps"])
        / baseline_metrics["speed_mae_mps"]
        if baseline_metrics["speed_mae_mps"] > 0.0 else None
    )

    keep_columns = [
        col for col in [
            "_Group_ID", "Source_dataset", "Flight_ID", "Timestamp", "Elapsed_s",
            "X", "Y", "Z", "Wind_speed", "Wind_angle"
        ] if col in engineered.columns
    ]
    predictions = engineered.iloc[test_idx][keep_columns].copy()
    predictions["Predicted_wind_speed"] = predicted_speed
    predictions["Predicted_wind_angle"] = predicted_angle
    predictions["Speed_error_mps"] = predicted_speed - speed_test
    predictions["Angle_error_deg"] = circular_difference_deg(angle_test, predicted_angle)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    per_flight = calculate_per_flight_metrics(predictions)
    per_flight.to_csv(output_dir / "per_flight_metrics.csv", index=False)
    metrics["flight_balanced_speed_mae_mps"] = float(per_flight["speed_mae_mps"].mean())
    metrics["flight_balanced_direction_circular_mae_deg"] = float(
        per_flight["direction_circular_mae_deg"].mean()
    )

    group_summary = engineered.groupby("_Group_ID", observed=True, sort=True).agg(
        Source_dataset=("Source_dataset", "first"),
        Flight_ID=("Flight_ID", "first"),
        rows=("_Group_ID", "size"),
    ).reset_index().rename(columns={"_Group_ID": "Group_ID"})
    group_summary["split"] = np.where(group_summary["Group_ID"].isin(train_groups), "train", "test")
    group_summary.to_csv(output_dir / "flight_split_manifest.csv", index=False)

    metrics.update({
        "script_version": SCRIPT_VERSION,
        "data_file": str(data_path.resolve()),
        "rows_total": len(engineered),
        "rows_train": len(train_idx),
        "rows_test": len(test_idx),
        "flights_total": unique_groups,
        "flights_train": len(train_groups),
        "flights_test": len(test_groups),
        "requested_train_size": train_size,
        "requested_test_size": test_size,
        "actual_train_row_fraction": len(train_idx) / len(engineered),
        "random_seed": args.random_seed,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "max_depth": max_depth,
        "max_features": args.max_features,
        "feature_count": len(feature_columns),
        "usable_time_fraction": feature_metadata["usable_time_fraction"],
        "time_features_enabled": feature_metadata["time_features_enabled"],
    })
    metrics = {key: json_value(value) for key, value in metrics.items()}

    artifact = {
        "script_version": SCRIPT_VERSION,
        "speed_model": speed_model,
        "direction_model": direction_model,
        "feature_columns": feature_columns,
        "random_seed": args.random_seed,
        "training_arguments": vars(args),
        "feature_metadata": feature_metadata,
        "notes": (
            "Direction is meteorological direction-from in degrees. "
            "Feature generation must preserve within-flight chronological row order."
        ),
    }
    joblib.dump(artifact, output_dir / "wind_model.joblib")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    if args.permutation_importance:
        sample_n = min(args.importance_sample_size, len(X_test))
        sample_rng = np.random.default_rng(args.random_seed)
        sample_positions = sample_rng.choice(len(X_test), size=sample_n, replace=False)
        X_sample = X_test.iloc[sample_positions]
        speed_sample = speed_test[sample_positions]
        direction_sample = direction_target[test_idx][sample_positions]

        speed_result = permutation_importance(
            speed_model,
            X_sample,
            speed_sample,
            n_repeats=3,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            scoring="neg_mean_absolute_error",
        )
        pd.DataFrame({
            "feature": feature_columns,
            "importance_mean": speed_result.importances_mean,
            "importance_std": speed_result.importances_std,
        }).sort_values("importance_mean", ascending=False).to_csv(
            output_dir / "feature_importance_speed.csv", index=False
        )

        direction_result = permutation_importance(
            direction_model,
            X_sample,
            direction_sample,
            n_repeats=3,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            scoring=direction_neg_circular_mae_scorer,
        )
        pd.DataFrame({
            "feature": feature_columns,
            "importance_mean": direction_result.importances_mean,
            "importance_std": direction_result.importances_std,
        }).sort_values("importance_mean", ascending=False).to_csv(
            output_dir / "feature_importance_direction.csv", index=False
        )

    if args.refit_all:
        all_flight_weights = equal_flight_weights(groups)
        if args.direction_min_speed > 0.0:
            all_direction_reliability = np.clip(speed_target / args.direction_min_speed, 0.05, 1.0)
        else:
            all_direction_reliability = np.ones_like(speed_target)
        all_direction_weights = all_flight_weights * all_direction_reliability
        all_direction_weights = all_direction_weights / np.mean(all_direction_weights)

        all_speed_model = make_regressor(
            args.n_estimators, args.min_samples_leaf, max_depth,
            args.max_features, args.n_jobs, args.random_seed,
        )
        all_direction_model = make_regressor(
            args.n_estimators, args.min_samples_leaf, max_depth,
            args.max_features, args.n_jobs, args.random_seed,
        )
        all_speed_model.fit(X, speed_target, model__sample_weight=all_flight_weights)
        all_direction_model.fit(X, direction_target, model__sample_weight=all_direction_weights)
        all_artifact = {
            **artifact,
            "speed_model": all_speed_model,
            "direction_model": all_direction_model,
            "trained_on_all_labeled_rows": True,
        }
        joblib.dump(all_artifact, output_dir / "wind_model_all_data.joblib")

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
