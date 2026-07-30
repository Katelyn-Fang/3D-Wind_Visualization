from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ARCHIVE_MARKERS = (
    "incomplete",
    "failed",
    "backup",
    "archive",
    "archived",
    "old",
)

MODEL_LABELS = {
    "dummy": "Dummy baseline",
    "ridge": "Ridge",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
    "lstm": "LSTM",
    "tcn": "TCN",
    "transformer": "Transformer",
}


@dataclass
class ModelPredictions:
    model: str
    display_name: str
    hardware: str
    path: Path
    frame: pd.DataFrame


def pretty_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())


def is_archived_path(path: Path) -> bool:
    return any(any(marker in part.lower() for marker in ARCHIVE_MARKERS) for part in path.parts)


def discover_prediction_files(results_dir: Path, expected_models: int = 12) -> list[Path]:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir.resolve()}")

    files = sorted(
        path for path in results_dir.rglob("test_predictions.csv")
        if not is_archived_path(path)
    )
    if expected_models > 0 and len(files) != expected_models:
        listed = "\n".join(f"  - {path}" for path in files)
        raise RuntimeError(
            f"Expected {expected_models} active test_predictions.csv files, found {len(files)}.\n"
            f"Archived/failed folders are excluded automatically.\n{listed}"
        )
    if not files:
        raise RuntimeError(f"No active test_predictions.csv files found under {results_dir.resolve()}")
    return files


def _read_available_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _pick_column(columns: Iterable[str], *candidates: str) -> str | None:
    columns = set(columns)
    return next((candidate for candidate in candidates if candidate in columns), None)


def _coerce_bool(series: pd.Series, default: bool = True) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(default)
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map({
        "true": True, "t": True, "1": True, "yes": True, "y": True,
        "false": False, "f": False, "0": False, "no": False, "n": False,
    })
    return mapped.fillna(default).astype(bool)


def _infer_model_name(path: Path, frame: pd.DataFrame) -> str:
    for column in ("Speed_model_name", "Direction_model_name", "Model", "model"):
        if column in frame.columns:
            values = frame[column].dropna().astype(str).str.strip()
            if not values.empty:
                return values.iloc[0].lower()
    folder = path.parent.name
    if "_" in folder and folder.split("_", 1)[0].isdigit():
        folder = folder.split("_", 1)[1]
    return folder.lower()


def _make_row_key(raw: pd.DataFrame, path: Path) -> tuple[pd.Series, pd.Series]:
    if "_Group_ID" in raw.columns:
        group = raw["_Group_ID"].astype("string").fillna("")
    elif {"Source_dataset", "Flight_ID"}.issubset(raw.columns):
        group = (
            raw["Source_dataset"].astype("string").fillna("unknown")
            + "::"
            + raw["Flight_ID"].astype("string").fillna("")
        )
    elif "Flight_ID" in raw.columns:
        group = raw["Flight_ID"].astype("string").fillna("")
    else:
        raise ValueError(f"{path}: need _Group_ID or Flight_ID to identify flights.")

    if "Sample_index" in raw.columns:
        sample = pd.to_numeric(raw["Sample_index"], errors="coerce")
        sample_text = sample.round().astype("Int64").astype("string")
    else:
        sample_text = raw.groupby(group, dropna=False).cumcount().astype("string")

    row_key = group + "||" + sample_text
    if row_key.duplicated().any():
        duplicate = row_key[row_key.duplicated()].iloc[0]
        raise ValueError(f"{path}: duplicate row key detected: {duplicate}")
    return group, row_key


def load_prediction_file(path: Path) -> ModelPredictions:
    path = Path(path)
    columns = _read_available_columns(path)

    required_candidates = {
        "Wind_speed", "Predicted_wind_speed", "Wind_angle", "Predicted_wind_angle",
        "_Group_ID", "Source_dataset", "Flight_ID", "Sample_index", "Timestamp",
        "Elapsed_s", "Common_sequence_eligible", "Predicted_direction_confidence",
        "Speed_model_name", "Direction_model_name",
    }
    usecols = [column for column in columns if column in required_candidates]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)

    required = ["Wind_speed", "Predicted_wind_speed", "Wind_angle", "Predicted_wind_angle"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing required prediction columns: {missing}")

    group, row_key = _make_row_key(raw, path)
    model = _infer_model_name(path, raw)
    hardware = "gpu" if "gpu" in {part.lower() for part in path.parts} else "cpu"

    frame = pd.DataFrame({
        "row_key": row_key,
        "group_id": group,
        "true_speed": pd.to_numeric(raw["Wind_speed"], errors="coerce"),
        "pred_speed": pd.to_numeric(raw["Predicted_wind_speed"], errors="coerce"),
        "true_angle": pd.to_numeric(raw["Wind_angle"], errors="coerce") % 360.0,
        "pred_angle": pd.to_numeric(raw["Predicted_wind_angle"], errors="coerce") % 360.0,
    })

    frame["sample_index"] = (
        pd.to_numeric(raw["Sample_index"], errors="coerce")
        if "Sample_index" in raw.columns
        else raw.groupby(group, dropna=False).cumcount().astype(float)
    )
    frame["timestamp"] = raw["Timestamp"].astype("string") if "Timestamp" in raw.columns else pd.Series(pd.NA, index=raw.index, dtype="string")
    frame["elapsed_s"] = pd.to_numeric(raw["Elapsed_s"], errors="coerce") if "Elapsed_s" in raw.columns else np.nan
    frame["eligible"] = _coerce_bool(raw["Common_sequence_eligible"]) if "Common_sequence_eligible" in raw.columns else True
    frame["confidence"] = (
        pd.to_numeric(raw["Predicted_direction_confidence"], errors="coerce")
        if "Predicted_direction_confidence" in raw.columns else np.nan
    )

    frame["speed_error"] = frame["pred_speed"] - frame["true_speed"]
    frame["abs_speed_error"] = frame["speed_error"].abs()
    frame["direction_error"] = ((frame["pred_angle"] - frame["true_angle"] + 180.0) % 360.0) - 180.0
    frame["abs_direction_error"] = frame["direction_error"].abs()

    return ModelPredictions(
        model=model,
        display_name=pretty_model_name(model),
        hardware=hardware,
        path=path,
        frame=frame,
    )


def load_all_predictions(results_dir: Path, expected_models: int = 12) -> dict[str, ModelPredictions]:
    loaded: dict[str, ModelPredictions] = {}
    for path in discover_prediction_files(results_dir, expected_models=expected_models):
        predictions = load_prediction_file(path)
        if predictions.model in loaded:
            raise RuntimeError(
                f"Duplicate active model name '{predictions.model}' found in:\n"
                f"  {loaded[predictions.model].path}\n  {predictions.path}"
            )
        loaded[predictions.model] = predictions
    return loaded


def common_row_keys(models: dict[str, ModelPredictions]) -> set[str]:
    common: set[str] | None = None
    for predictions in models.values():
        frame = predictions.frame
        valid = (
            frame["eligible"]
            & frame["true_speed"].notna()
            & frame["pred_speed"].notna()
            & frame["true_angle"].notna()
            & frame["pred_angle"].notna()
        )
        keys = set(frame.loc[valid, "row_key"].astype(str))
        common = keys if common is None else common.intersection(keys)
    if not common:
        raise RuntimeError("No common eligible prediction rows exist across all active models.")
    return common


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 2:
        return float("nan")
    xv = x[valid].to_numpy(dtype=float)
    yv = y[valid].to_numpy(dtype=float)
    if np.std(xv) == 0.0 or np.std(yv) == 0.0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def compute_common_comparison(
    models: dict[str, ModelPredictions],
    direction_min_speed: float = 1.0,
    speed_weight: float = 0.5,
    direction_weight: float = 0.5,
) -> tuple[pd.DataFrame, set[str]]:
    if direction_min_speed < 0:
        raise ValueError("direction_min_speed must be nonnegative.")
    if speed_weight < 0 or direction_weight < 0 or speed_weight + direction_weight <= 0:
        raise ValueError("speed and direction weights must be nonnegative and sum to a positive value.")

    speed_weight = speed_weight / (speed_weight + direction_weight)
    direction_weight = 1.0 - speed_weight
    keys = common_row_keys(models)
    rows: list[dict[str, object]] = []

    for model, predictions in models.items():
        frame = predictions.frame[predictions.frame["row_key"].isin(keys)].copy()
        direction = frame[frame["true_speed"] > direction_min_speed].copy()
        rows.append({
            "model": model,
            "display_name": predictions.display_name,
            "hardware": predictions.hardware,
            "prediction_file": str(predictions.path),
            "common_n": int(len(frame)),
            "common_speed_mae_mps": float(frame["abs_speed_error"].mean()),
            "common_speed_rmse_mps": float(np.sqrt(np.mean(np.square(frame["speed_error"])))),
            "common_speed_r": pearson_r(frame["true_speed"], frame["pred_speed"]),
            "direction_n_above_threshold": int(len(direction)),
            "common_direction_mae_deg_above_threshold": float(direction["abs_direction_error"].mean()),
            "common_direction_median_abs_error_deg_above_threshold": float(direction["abs_direction_error"].median()),
            "common_direction_p90_abs_error_deg_above_threshold": float(direction["abs_direction_error"].quantile(0.90)),
            "mean_direction_confidence_above_threshold": float(direction["confidence"].mean()),
            "direction_speed_threshold_mps": float(direction_min_speed),
        })

    comparison = pd.DataFrame(rows)
    if comparison["common_speed_mae_mps"].isna().any():
        raise RuntimeError("At least one model has a missing common speed MAE.")
    if comparison["common_direction_mae_deg_above_threshold"].isna().any():
        raise RuntimeError("At least one model has a missing direction MAE above the threshold.")

    best_speed = comparison["common_speed_mae_mps"].min()
    best_direction = comparison["common_direction_mae_deg_above_threshold"].min()
    comparison["speed_regret_vs_best"] = comparison["common_speed_mae_mps"] / best_speed
    comparison["direction_regret_vs_best"] = comparison["common_direction_mae_deg_above_threshold"] / best_direction
    comparison["balanced_hybrid_score"] = np.exp(
        speed_weight * np.log(comparison["speed_regret_vs_best"])
        + direction_weight * np.log(comparison["direction_regret_vs_best"])
    )
    comparison["balanced_worst_regret"] = comparison[["speed_regret_vs_best", "direction_regret_vs_best"]].max(axis=1)
    comparison["speed_rank"] = comparison["common_speed_mae_mps"].rank(method="min").astype(int)
    comparison["direction_rank"] = comparison["common_direction_mae_deg_above_threshold"].rank(method="min").astype(int)
    comparison["balanced_rank"] = comparison["balanced_hybrid_score"].rank(method="min").astype(int)

    comparison = comparison.sort_values(
        ["balanced_hybrid_score", "balanced_worst_regret", "common_speed_mae_mps", "model"]
    ).reset_index(drop=True)
    return comparison, keys


def select_winners(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    speed_row = comparison.sort_values(
        ["common_speed_mae_mps", "common_direction_mae_deg_above_threshold", "model"]
    ).iloc[0]
    direction_row = comparison.sort_values(
        ["common_direction_mae_deg_above_threshold", "common_speed_mae_mps", "model"]
    ).iloc[0]
    balanced_row = comparison.sort_values(
        ["balanced_hybrid_score", "balanced_worst_regret", "common_speed_mae_mps", "model"]
    ).iloc[0]

    selections = pd.DataFrame([
        {
            "category": "best_common_speed",
            "model": speed_row["model"],
            "display_name": speed_row["display_name"],
            "metric": "common_speed_mae_mps",
            "value": speed_row["common_speed_mae_mps"],
            "units": "m/s",
            "definition": "Lowest speed MAE on the exact common eligible rows shared by all models.",
        },
        {
            "category": "best_direction_above_threshold",
            "model": direction_row["model"],
            "display_name": direction_row["display_name"],
            "metric": "common_direction_mae_deg_above_threshold",
            "value": direction_row["common_direction_mae_deg_above_threshold"],
            "units": "degrees",
            "definition": (
                "Lowest circular direction MAE on common rows whose true wind speed is above "
                f"{direction_row['direction_speed_threshold_mps']:g} m/s."
            ),
        },
        {
            "category": "best_balanced_hybrid",
            "model": balanced_row["model"],
            "display_name": balanced_row["display_name"],
            "metric": "balanced_hybrid_score",
            "value": balanced_row["balanced_hybrid_score"],
            "units": "relative score; lower is better",
            "definition": (
                "Equal-weight geometric mean of speed-MAE regret and direction-MAE regret versus "
                "the best model in each metric. This selects a joint compromise; it is not an ensemble."
            ),
        },
    ])

    roles = selections.groupby("model")["category"].agg(lambda values: ";".join(values)).to_dict()
    finalists = comparison[comparison["model"].isin(roles)].copy()
    finalists["roles"] = finalists["model"].map(roles)
    finalists = finalists.sort_values(["balanced_rank", "speed_rank", "direction_rank"]).reset_index(drop=True)
    return selections, finalists


def common_frames(
    models: dict[str, ModelPredictions],
    keys: set[str],
) -> dict[str, pd.DataFrame]:
    return {
        model: predictions.frame[predictions.frame["row_key"].isin(keys)].copy()
        for model, predictions in models.items()
    }
