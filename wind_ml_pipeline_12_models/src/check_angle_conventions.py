#!/usr/bin/env python3
"""Compare common yaw conventions against GPS/course direction on moving telemetry rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wind_core import (
    add_engineered_features,
    circular_difference_deg,
    load_standardized_data,
    yaw_to_heading_deg,
)

TRANSFORMS = [
    "clockwise_from_north",
    "counterclockwise_from_north",
    "ccw_from_east_to_heading",
    "cw_from_east_to_heading",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--attitude-angle-unit", choices=["auto", "radians", "degrees"], default="auto"
    )
    parser.add_argument(
        "--minimum-horizontal-step-m",
        type=float,
        default=0.25,
        help="Ignore nearly stationary rows where GPS course is unstable.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data = load_standardized_data(Path(args.data))
    engineered, _, metadata = add_engineered_features(
        data, attitude_angle_unit=args.attitude_angle_unit
    )
    group = engineered.groupby("_Group_ID", sort=False, observed=True)
    east_step = group["Local_east_m"].diff().to_numpy(dtype=float)
    north_step = group["Local_north_m"].diff().to_numpy(dtype=float)
    step = np.sqrt(east_step**2 + north_step**2)
    course = (np.degrees(np.arctan2(east_step, north_step)) + 360.0) % 360.0
    valid = (
        np.isfinite(course)
        & np.isfinite(step)
        & (step >= args.minimum_horizontal_step_m)
        & np.isfinite(engineered["Yaw"].to_numpy(dtype=float))
    )
    if not valid.any():
        raise ValueError(
            "No moving rows passed the course filter. Reduce --minimum-horizontal-step-m "
            "or inspect the position columns."
        )

    rows = []
    yaw = engineered["Yaw"].to_numpy(dtype=float)
    for transform in TRANSFORMS:
        heading = yaw_to_heading_deg(
            yaw,
            metadata["attitude_angle_unit_resolved"],
            transform,
        )
        error = circular_difference_deg(course[valid], heading[valid])
        rows.append(
            {
                "yaw_transform": transform,
                "rows": int(valid.sum()),
                "course_heading_circular_mae_deg": float(np.mean(np.abs(error))),
                "course_heading_circular_rmse_deg": float(np.sqrt(np.mean(error**2))),
                "within_30deg_fraction": float(np.mean(np.abs(error) <= 30.0)),
            }
        )

    results = pd.DataFrame(rows).sort_values("course_heading_circular_mae_deg")
    print(f"Resolved attitude unit: {metadata['attitude_angle_unit_resolved']}")
    print(results.to_string(index=False))
    print(
        "\nInterpretation: this is a diagnostic, not proof. Drone yaw can differ from GPS "
        "course during hovering, side-slip, or turns. Enable relative_yaw only when the best "
        "transform is clearly superior and agrees with the dataset documentation."
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
        print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
