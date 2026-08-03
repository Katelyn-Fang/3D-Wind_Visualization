"""Create a standalone measured/baseline/physics/ML wind comparison graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def physics_direction_from(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.mod(np.rad2deg(np.arctan2(-u, -v)), 360.0)


def select_flight(frame: pd.DataFrame, requested: str | None) -> pd.DataFrame:
    flight = requested or str(frame["Flight_ID"].value_counts().index[0])
    selected = frame.loc[frame["Flight_ID"].astype(str) == flight].copy()
    if selected.empty:
        examples = ", ".join(map(str, frame["Flight_ID"].drop_duplicates()[:8]))
        raise ValueError(f"Flight {flight!r} not found. Examples: {examples}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight", help="Flight_ID; defaults to the flight with the most rows")
    parser.add_argument("--start", type=int, default=0, help="First row within the flight")
    parser.add_argument("--count", type=int, default=1500, help="Maximum rows to plot")
    parser.add_argument("--output", type=Path, default=ROOT / "wind_model_comparison.png")
    parser.add_argument("--show", action="store_true", help="Open the graph after saving it")
    args = parser.parse_args()

    ml = pd.read_csv(ROOT / "test_predictions.csv").reset_index(names="source_index")
    physics = pd.read_csv(ROOT / "physics_test_predictions.csv")
    if len(ml) != len(physics):
        raise ValueError("ML and physics files must contain the same held-out rows")

    selected = select_flight(ml, args.flight)
    start = max(args.start, 0)
    selected = selected.iloc[start:start + max(args.count, 1)].copy()
    if selected.empty:
        raise ValueError("The selected start is beyond the end of this flight")
    physics_selected = physics.iloc[selected["source_index"].to_numpy()].reset_index(drop=True)
    selected = selected.reset_index(drop=True)

    # Persistence baseline: use the previous measured value within the flight.
    baseline_speed = selected["Wind_speed"].shift(1)
    baseline_direction = selected["Wind_angle"].shift(1)
    physics_direction = physics_direction_from(
        physics_selected["Physics_u"].to_numpy(float),
        physics_selected["Physics_v"].to_numpy(float),
    )
    sample = np.arange(len(selected))

    figure, (speed_axis, direction_axis) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, constrained_layout=True
    )
    figure.suptitle(
        f"Observed and predicted wind — flight {selected['Flight_ID'].iloc[0]}",
        fontsize=14,
    )
    styles = [
        ("Measured", "#405cff", "-", 1.8),
        ("Baseline", "#8a8f98", ":", 1.3),
        ("Physics", "#ff8c32", "--", 1.5),
        ("Machine learning", "#00a881", "-.", 1.5),
    ]
    speed_values = [
        selected["Wind_speed"], baseline_speed,
        physics_selected["Physics_speed"], selected["Predicted_wind_speed"],
    ]
    direction_values = [
        selected["Wind_angle"], baseline_direction,
        physics_direction, selected["Predicted_wind_angle"],
    ]

    for (label, color, line_style, width), values in zip(styles, speed_values):
        speed_axis.plot(sample, values, label=label, color=color, linestyle=line_style, linewidth=width)
    speed_axis.set(title="Wind speed", ylabel="Speed (m/s)")
    speed_axis.set_ylim(bottom=0)
    speed_axis.grid(alpha=0.25)
    speed_axis.legend(ncol=4, loc="upper center", frameon=False)

    for (label, color, line_style, width), values in zip(styles, direction_values):
        direction_axis.plot(sample, values, label=label, color=color, linestyle=line_style, linewidth=width)
    direction_axis.set(
        title="Wind direction-from", ylabel="Direction (°)", xlabel="Sample", ylim=(0, 360)
    )
    direction_axis.set_yticks([0, 90, 180, 270, 360])
    direction_axis.grid(alpha=0.25)
    direction_axis.legend(ncol=4, loc="upper center", frameon=False)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"Saved {output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
