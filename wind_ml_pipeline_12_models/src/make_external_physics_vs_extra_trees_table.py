#!/usr/bin/env python3
"""Compare physics-baseline and Extra Trees predictions on external datasets.

The repository contains ground truth and Extra Trees predictions, but not the
row-level physics-baseline predictions. Supply one physics CSV per dataset.
Each physics CSV must contain a row key plus predicted speed and direction.
Run ``python src/make_external_physics_vs_extra_trees_table.py --help`` for use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AMOVFLY = ROOT / "results/external_amovfly/extra_trees/test_predictions.csv"
DEFAULT_DRONE = ROOT / "results/external_drone_onboard/predictions/extra_trees.csv"
DEFAULT_OUTPUT = ROOT / "results/scc/final_analysis/poster_regenerated"

SPEED_PREDICTION_NAMES = [
    "Physics_wind_speed", "physics_wind_speed", "Predicted_wind_speed",
    "predicted_wind_speed", "pred_speed", "wind_speed_prediction",
]
DIRECTION_PREDICTION_NAMES = [
    "Physics_wind_angle", "physics_wind_angle", "Predicted_wind_angle",
    "predicted_wind_angle", "pred_angle", "wind_angle_prediction",
]
ROW_KEY_NAMES = ["_External_row_id", "row_id", "Row_ID", "Sample_index"]
GROUP_NAMES = ["_Group_ID", "group_id", "Group_ID"]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create a poster table comparing physics and Extra Trees against ground truth."
    )
    p.add_argument("--physics-amovfly", type=Path, required=True)
    p.add_argument("--physics-drone-onboard", type=Path, required=True)
    p.add_argument("--extra-trees-amovfly", type=Path, default=DEFAULT_AMOVFLY)
    p.add_argument("--extra-trees-drone-onboard", type=Path, default=DEFAULT_DRONE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--direction-min-speed", type=float, default=1.0)
    return p


def first_present(columns, choices: list[str], label: str) -> str:
    for name in choices:
        if name in columns:
            return name
    raise KeyError(f"Could not find {label}. Accepted names: {choices}")


def normalize_extra_trees(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    group = first_present(frame.columns, GROUP_NAMES, "group identifier")
    result = pd.DataFrame({
        "group_id": frame[group].astype(str),
        "true_speed": pd.to_numeric(frame["Wind_speed"], errors="coerce"),
        "true_angle": pd.to_numeric(frame["Wind_angle"], errors="coerce"),
        "extra_trees_speed": pd.to_numeric(frame["Predicted_wind_speed"], errors="coerce"),
        "extra_trees_angle": pd.to_numeric(frame["Predicted_wind_angle"], errors="coerce"),
    })
    row_key = next((name for name in ROW_KEY_NAMES if name in frame.columns), None)
    if row_key:
        result["row_key"] = frame[row_key].astype(str)
    else:
        # Deterministic within-flight row number is the fallback alignment key.
        result["row_key"] = frame.groupby(group, sort=False).cumcount().astype(str)
    return result


def normalize_physics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    group = first_present(frame.columns, GROUP_NAMES, "group identifier")
    speed = first_present(frame.columns, SPEED_PREDICTION_NAMES, "physics speed prediction")
    angle = first_present(frame.columns, DIRECTION_PREDICTION_NAMES, "physics direction prediction")
    result = pd.DataFrame({
        "group_id": frame[group].astype(str),
        "physics_speed": pd.to_numeric(frame[speed], errors="coerce"),
        "physics_angle": pd.to_numeric(frame[angle], errors="coerce"),
    })
    row_key = next((name for name in ROW_KEY_NAMES if name in frame.columns), None)
    if row_key:
        result["row_key"] = frame[row_key].astype(str)
    else:
        result["row_key"] = frame.groupby(group, sort=False).cumcount().astype(str)
    if result.duplicated(["group_id", "row_key"]).any():
        raise ValueError(f"Physics file has duplicate alignment keys: {path}")
    return result


def circular_absolute_error(predicted, observed) -> np.ndarray:
    return np.abs((np.asarray(predicted) - np.asarray(observed) + 180.0) % 360.0 - 180.0)


def metrics(dataset: str, method: str, frame: pd.DataFrame, speed_col: str,
            angle_col: str, threshold: float) -> dict[str, object]:
    speed_error = frame[speed_col] - frame["true_speed"]
    direction_rows = frame[frame["true_speed"] > threshold]
    angle_error = circular_absolute_error(direction_rows[angle_col], direction_rows["true_angle"])
    return {
        "Dataset": dataset,
        "Method": method,
        "Rows": len(frame),
        "Flights": frame["group_id"].nunique(),
        "Speed MAE (m/s)": speed_error.abs().mean(),
        "Speed RMSE (m/s)": np.sqrt(np.mean(speed_error**2)),
        f"Direction MAE > {threshold:g} m/s (deg)": np.mean(angle_error),
        f"Within 15 deg > {threshold:g} m/s (%)": 100.0 * np.mean(angle_error <= 15.0),
    }


def evaluate(dataset: str, extra_path: Path, physics_path: Path,
             threshold: float) -> tuple[pd.DataFrame, dict[str, int]]:
    extra = normalize_extra_trees(extra_path)
    physics = normalize_physics(physics_path)
    joined = extra.merge(
        physics, on=["group_id", "row_key"], how="inner", validate="one_to_one"
    ).dropna(subset=[
        "true_speed", "true_angle", "extra_trees_speed", "extra_trees_angle",
        "physics_speed", "physics_angle",
    ])
    if joined.empty:
        raise RuntimeError(f"No aligned valid rows for {dataset}")
    coverage = {
        "extra_trees_rows": len(extra), "physics_rows": len(physics),
        "aligned_rows": len(joined),
    }
    table = pd.DataFrame([
        metrics(dataset, "Physics baseline", joined, "physics_speed", "physics_angle", threshold),
        metrics(dataset, "Extra Trees", joined, "extra_trees_speed", "extra_trees_angle", threshold),
    ])
    physics_speed = table.loc[table["Method"] == "Physics baseline", "Speed MAE (m/s)"].iloc[0]
    extra_speed = table.loc[table["Method"] == "Extra Trees", "Speed MAE (m/s)"].iloc[0]
    table["Speed improvement vs physics (%)"] = [0.0, 100.0 * (physics_speed - extra_speed) / physics_speed]
    return table, coverage


def draw_table(table: pd.DataFrame, output: Path, threshold: float) -> None:
    compact = table[[
        "Dataset", "Method", "Speed MAE (m/s)",
        f"Direction MAE > {threshold:g} m/s (deg)",
        f"Within 15 deg > {threshold:g} m/s (%)",
        "Speed improvement vs physics (%)",
    ]].copy()
    compact.columns = [
        "Dataset", "Method", "Speed MAE\n(m/s)", "Direction MAE\n(deg)",
        "Within 15 deg\n(%)", "Speed improvement\nvs physics (%)",
    ]
    for col in compact.columns[2:]:
        compact[col] = compact[col].map(lambda value: f"{value:.1f}")

    fig, ax = plt.subplots(figsize=(11.2, 2.35))
    ax.axis("off")
    rendered = ax.table(
        cellText=compact.values, colLabels=compact.columns,
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.16, 0.18, 0.14, 0.17, 0.16, 0.19],
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.65)
    for (row, col), cell in rendered.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#315F9B")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF3F8")
    fig.suptitle("External Performance Relative to Ground Truth", fontsize=14, weight="bold")
    fig.text(
        0.5, 0.035,
        f"All methods are evaluated on identical aligned rows. Direction metrics use true wind speed > {threshold:g} m/s; lower MAE is better.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=[0.01, 0.07, 0.99, 0.9])
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    amov, amov_coverage = evaluate(
        "AMOVFLY", args.extra_trees_amovfly, args.physics_amovfly,
        args.direction_min_speed,
    )
    drone, drone_coverage = evaluate(
        "Drone Onboard", args.extra_trees_drone_onboard,
        args.physics_drone_onboard, args.direction_min_speed,
    )
    table = pd.concat([amov, drone], ignore_index=True)
    table.to_csv(args.output_dir / "external_physics_vs_extra_trees_metrics.csv", index=False)
    pd.DataFrame([
        {"Dataset": "AMOVFLY", **amov_coverage},
        {"Dataset": "Drone Onboard", **drone_coverage},
    ]).to_csv(args.output_dir / "external_physics_alignment_audit.csv", index=False)
    draw_table(
        table,
        args.output_dir / "external_physics_vs_extra_trees_poster_table",
        args.direction_min_speed,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
