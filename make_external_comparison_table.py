#!/usr/bin/env python3
"""Create a measured/Extra Trees/physics comparison table.

The Extra Trees and physics files must describe the same measured samples in
the same order. This keeps the comparison fair and prevents metrics from two
different flights or row subsets from being combined.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    """Return the first available column name from a list of aliases."""
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def required_column(
    frame: pd.DataFrame,
    names: tuple[str, ...],
    description: str,
) -> str:
    column = first_column(frame, names)
    if column is None:
        raise KeyError(
            f"Could not find {description}. Tried: {', '.join(names)}"
        )
    return column


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def circular_difference(predicted: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """Signed angular difference in the interval [-180, 180)."""
    return (predicted - measured + 180.0) % 360.0 - 180.0


def circular_mean(degrees: np.ndarray) -> float:
    radians = np.deg2rad(degrees)
    return float(
        np.rad2deg(
            np.arctan2(
                np.mean(np.sin(radians)),
                np.mean(np.cos(radians)),
            )
        )
    )


def direction_from_components(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Convert east/north wind-to components to meteorological direction-from."""
    return np.mod(np.rad2deg(np.arctan2(-u, -v)), 360.0)


def model_metrics(
    name: str,
    measured_speed: np.ndarray,
    measured_direction: np.ndarray,
    predicted_speed: np.ndarray,
    predicted_direction: np.ndarray,
    direction_min_speed: float,
) -> dict[str, float | int | str]:
    valid_speed = np.isfinite(measured_speed) & np.isfinite(predicted_speed)
    speed_mae = float(
        np.mean(np.abs(predicted_speed[valid_speed] - measured_speed[valid_speed]))
    )

    angle_error = circular_difference(predicted_direction, measured_direction)
    valid_direction = (
        np.isfinite(angle_error)
        & np.isfinite(measured_speed)
        & np.isfinite(predicted_speed)
        & (measured_speed > direction_min_speed)
        & (predicted_speed >= 0.2)
    )
    reliable_error = angle_error[valid_direction]
    if not len(reliable_error):
        raise ValueError(f"No reliable direction rows were available for {name}")

    return {
        "Model": name,
        "Speed MAE (m/s)": speed_mae,
        "Direction MAE (deg)": float(np.mean(np.abs(reliable_error))),
        "Angular offset (deg)": circular_mean(reliable_error),
        "Within 15 deg (%)": float(np.mean(np.abs(reliable_error) <= 15.0) * 100.0),
        "Compared rows": int(valid_speed.sum()),
    }


def load_predictions(
    extra_trees_path: Path,
    physics_path: Path,
    direction_min_speed: float,
) -> pd.DataFrame:
    extra = pd.read_csv(extra_trees_path, low_memory=False)
    physics = pd.read_csv(physics_path, low_memory=False)

    if len(extra) != len(physics):
        raise ValueError(
            "The files do not contain the same number of rows: "
            f"Extra Trees has {len(extra):,}; physics has {len(physics):,}. "
            "Generate both predictions on the same measured samples."
        )

    measured_speed_column = required_column(
        extra,
        ("Wind_speed", "wind_speed", "measured_speed", "true_speed"),
        "measured wind speed in the Extra Trees file",
    )
    measured_direction_column = required_column(
        extra,
        ("Wind_angle", "wind_angle", "measured_direction", "true_angle"),
        "measured wind direction in the Extra Trees file",
    )
    extra_speed_column = required_column(
        extra,
        ("Predicted_wind_speed", "predicted_wind_speed", "pred_speed"),
        "Extra Trees predicted speed",
    )
    extra_direction_column = required_column(
        extra,
        ("Predicted_wind_angle", "predicted_wind_angle", "pred_angle"),
        "Extra Trees predicted direction",
    )

    measured_speed = numeric(extra, measured_speed_column)
    measured_direction = numeric(extra, measured_direction_column)

    # If the physics file includes measured values, verify row alignment.
    physics_measured_speed_column = first_column(
        physics,
        ("Wind_speed", "wind_speed", "measured_speed", "true_speed"),
    )
    physics_measured_direction_column = first_column(
        physics,
        ("Wind_angle", "wind_angle", "measured_direction", "true_angle"),
    )
    if physics_measured_speed_column:
        physics_measured_speed = numeric(physics, physics_measured_speed_column)
        if not np.allclose(
            measured_speed,
            physics_measured_speed,
            equal_nan=True,
            atol=1e-6,
        ):
            raise ValueError("Measured wind speeds do not align between the files")
    if physics_measured_direction_column:
        physics_measured_direction = numeric(physics, physics_measured_direction_column)
        mismatch = np.abs(
            circular_difference(physics_measured_direction, measured_direction)
        )
        if np.nanmax(mismatch) > 1e-6:
            raise ValueError("Measured wind directions do not align between the files")

    physics_speed_column = first_column(
        physics,
        ("Physics_speed", "physics_speed", "predicted_speed"),
    )
    physics_u_column = first_column(physics, ("Physics_u", "physics_u", "u"))
    physics_v_column = first_column(physics, ("Physics_v", "physics_v", "v"))
    if physics_speed_column:
        physics_speed = numeric(physics, physics_speed_column)
    elif physics_u_column and physics_v_column:
        physics_speed = np.hypot(
            numeric(physics, physics_u_column),
            numeric(physics, physics_v_column),
        )
    else:
        raise KeyError("Physics file needs Physics_speed or Physics_u and Physics_v")

    physics_direction_column = first_column(
        physics,
        ("Physics_direction", "physics_direction", "predicted_direction"),
    )
    if physics_direction_column:
        physics_direction = numeric(physics, physics_direction_column)
    elif physics_u_column and physics_v_column:
        physics_direction = direction_from_components(
            numeric(physics, physics_u_column),
            numeric(physics, physics_v_column),
        )
    else:
        raise KeyError(
            "Physics file needs Physics_direction or Physics_u and Physics_v"
        )

    rows = [
        {
            "Model": "Measured",
            "Speed MAE (m/s)": 0.0,
            "Direction MAE (deg)": 0.0,
            "Angular offset (deg)": 0.0,
            "Within 15 deg (%)": 100.0,
            "Compared rows": int(np.isfinite(measured_speed).sum()),
        },
        model_metrics(
            "Extra Trees",
            measured_speed,
            measured_direction,
            numeric(extra, extra_speed_column),
            numeric(extra, extra_direction_column),
            direction_min_speed,
        ),
        model_metrics(
            "Physics",
            measured_speed,
            measured_direction,
            physics_speed,
            physics_direction,
            direction_min_speed,
        ),
    ]
    return pd.DataFrame(rows)


def save_table(metrics: pd.DataFrame, dataset: str, output: Path) -> None:
    display = metrics.copy()
    for column in (
        "Speed MAE (m/s)",
        "Direction MAE (deg)",
        "Angular offset (deg)",
        "Within 15 deg (%)",
    ):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    display["Compared rows"] = display["Compared rows"].map(lambda value: f"{value:,}")
    display = display.rename(
        columns={
            "Speed MAE (m/s)": "Speed MAE\n(m/s)",
            "Direction MAE (deg)": "Direction MAE\n(degrees)",
            "Angular offset (deg)": "Angular offset\n(degrees)",
            "Within 15 deg (%)": "Within 15°\n(%)",
            "Compared rows": "Samples",
        }
    )

    figure, axis = plt.subplots(figsize=(12.5, 2.7), facecolor="white")
    axis.set_facecolor("white")
    axis.axis("off")
    axis.set_title(
        f"{dataset}: measured wind compared with Extra Trees and physics",
        fontsize=14,
        weight="bold",
        pad=14,
    )
    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        colWidths=[0.18, 0.15, 0.18, 0.18, 0.15, 0.13],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.8)
    for (row, _column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(color="black", weight="bold")

    figure.text(
        0.5,
        0.035,
        "Direction metrics use measured wind speed > 1 m/s. "
        "Angular offset near 0 degrees is best.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an external Extra Trees versus physics comparison table."
    )
    parser.add_argument("--extra-trees", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--dataset", default="External dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("external_model_comparison_table.png"),
    )
    parser.add_argument("--direction-min-speed", type=float, default=1.0)
    args = parser.parse_args()

    metrics = load_predictions(
        args.extra_trees,
        args.physics,
        args.direction_min_speed,
    )
    save_table(metrics, args.dataset, args.output)
    csv_output = args.output.with_suffix(".csv")
    metrics.round(3).to_csv(csv_output, index=False)
    print(metrics.round(3).to_string(index=False))
    print(f"\nSaved table image: {args.output.resolve()}")
    print(f"Saved table data:  {csv_output.resolve()}")


if __name__ == "__main__":
    main()
