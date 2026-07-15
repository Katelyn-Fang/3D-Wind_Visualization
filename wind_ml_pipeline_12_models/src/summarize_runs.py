#!/usr/bin/env python3
"""Combine model_metrics.csv files and build full/common-endpoint rankings."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.results_dir)
    files = sorted(root.rglob("model_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No model_metrics.csv files found under {root.resolve()}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame.insert(0, "run_directory", str(path.parent))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output = Path(args.output) if args.output else root / "model_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    full_sort = [c for c in ["flight_balanced_speed_mae_mps", "flight_balanced_direction_circular_mae_deg"] if c in combined]
    full = combined.sort_values(full_sort) if full_sort else combined
    full.to_csv(root / "model_ranking_full_scope.csv", index=False)

    common_sort = [c for c in ["common_flight_balanced_speed_mae_mps", "common_flight_balanced_direction_circular_mae_deg"] if c in combined]
    common = combined.dropna(subset=common_sort, how="any") if common_sort else combined.iloc[0:0]
    if len(common):
        common = common.sort_values(common_sort)
        common.to_csv(root / "model_ranking_common_endpoints.csv", index=False)

    direction_primary = "common_flight_balanced_direction_speed_ge_1p0_mps_circular_mae_deg"
    direction_secondary = "common_flight_balanced_direction_speed_ge_2p0_mps_circular_mae_deg"
    if direction_primary in combined.columns:
        direction = combined.dropna(subset=[direction_primary]).copy()
        direction["direction_rank_ge_1mps"] = direction[direction_primary].rank(
            method="min", ascending=True
        )
        if direction_secondary in direction.columns:
            direction["direction_rank_ge_2mps"] = direction[direction_secondary].rank(
                method="min", ascending=True
            )
        direction = direction.sort_values(
            [column for column in [direction_primary, direction_secondary] if column in direction]
        )
        direction.to_csv(root / "model_ranking_direction.csv", index=False)

    speed_primary = "common_flight_balanced_speed_mae_mps"
    if speed_primary in combined.columns and direction_primary in combined.columns:
        balanced = combined.dropna(subset=[speed_primary, direction_primary]).copy()
        balanced["speed_rank"] = balanced[speed_primary].rank(method="min", ascending=True)
        balanced["direction_rank_ge_1mps"] = balanced[direction_primary].rank(
            method="min", ascending=True
        )
        balanced["balanced_rank_score"] = (
            balanced["speed_rank"] + balanced["direction_rank_ge_1mps"]
        ) / 2.0
        balanced.sort_values("balanced_rank_score").to_csv(
            root / "model_ranking_balanced.csv", index=False
        )

    display = [c for c in [
        "model_name", "model_family", "evaluation_scope", "rows_test",
        "speed_mae_mps", "direction_all_circular_mae_deg",
        "common_flight_balanced_speed_mae_mps",
        "common_flight_balanced_direction_circular_mae_deg",
        "common_flight_balanced_direction_speed_ge_1p0_mps_circular_mae_deg",
        "common_flight_balanced_direction_speed_ge_2p0_mps_circular_mae_deg",
        "training_seconds",
    ] if c in combined]
    print(combined[display].sort_values("model_name").to_string(index=False))
    print(f"Saved comparison: {output.resolve()}")


if __name__ == "__main__":
    main()
