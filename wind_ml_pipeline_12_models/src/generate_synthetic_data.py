#!/usr/bin/env python3
"""Generate a small learnable telemetry dataset for software smoke tests only."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def generate_dataset(n_flights: int, rows_per_flight: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-07-01 09:00:00")
    for flight in range(n_flights):
        t = np.arange(rows_per_flight, dtype=float)
        phase = rng.uniform(0, 2 * np.pi)
        if flight % 4 == 0:
            base_angle = 355.0
        elif flight % 4 == 1:
            base_angle = 5.0
        else:
            base_angle = rng.uniform(0, 360)
        base_speed = rng.uniform(1.0, 4.0)
        east = np.cumsum(0.6 + 0.15 * np.sin(t / 8 + phase) + rng.normal(0, 0.05, rows_per_flight))
        north = np.cumsum(0.4 + 0.12 * np.cos(t / 10 + phase) + rng.normal(0, 0.05, rows_per_flight))
        altitude = 12 + 0.04 * t + 1.5 * np.sin(t / 14 + phase)
        wind_speed = (
            base_speed
            + 0.55 * np.sin(t / 11 + phase)
            + 0.25 * np.cos(t / 5)
            + rng.normal(0, 0.10, rows_per_flight)
        )
        wind_speed = np.maximum(wind_speed, 0.1)
        if flight % 3 == 0:
            calm_start = rows_per_flight // 3
            calm_stop = min(rows_per_flight, calm_start + max(5, rows_per_flight // 8))
            wind_speed[calm_start:calm_stop] = np.maximum(
                0.15,
                0.35 + rng.normal(0, 0.05, calm_stop - calm_start),
            )
        wind_angle = (base_angle + 22 * np.sin(t / 18 + phase) + 0.35 * t) % 360
        angle_rad = np.deg2rad(wind_angle)
        roll = 0.06 * wind_speed * np.sin(angle_rad) + rng.normal(0, 0.015, rows_per_flight)
        pitch = 0.06 * wind_speed * np.cos(angle_rad) + rng.normal(0, 0.015, rows_per_flight)
        yaw = np.unwrap(angle_rad + 0.12 * np.sin(t / 9))
        # Wrap to [-pi, pi) like real attitude telemetry; unwrapped radians can
        # exceed the auto unit-detection threshold and be misread as degrees.
        yaw = np.mod(yaw + np.pi, 2.0 * np.pi) - np.pi
        battery_v = 16.8 - 0.0045 * t - 0.02 * flight + rng.normal(0, 0.01, rows_per_flight)
        battery_c = 5.0 + 1.4 * wind_speed + 7.0 * np.sqrt(roll**2 + pitch**2) + rng.normal(0, 0.10, rows_per_flight)
        for i in range(rows_per_flight):
            rows.append(
                {
                    "Timestamp": start + pd.Timedelta(minutes=flight * 20, seconds=int(i)),
                    "Elapsed_s": float(i),
                    "X": float(east[i]),
                    "Y": float(north[i]),
                    "Z": float(altitude[i]),
                    "Roll": float(roll[i]),
                    "Pitch": float(pitch[i]),
                    "Yaw": float(yaw[i]),
                    "Battery_V": float(battery_v[i]),
                    "Battery_C": float(battery_c[i]),
                    "Wind_speed": float(wind_speed[i]),
                    "Wind_angle": float(wind_angle[i]),
                    "Source_dataset": "synthetic_smoke_test",
                    "Flight_ID": f"flight_{flight:02d}",
                    "Coordinate_frame": "cartesian_meters",
                }
            )
    return pd.DataFrame(rows)


def write_manifest(df: pd.DataFrame, path: Path, test_size: float, seed: int) -> None:
    groups = np.array(sorted((df["Source_dataset"] + "::" + df["Flight_ID"]).unique()))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    positions = np.arange(len(groups))
    train_pos, test_pos = next(splitter.split(positions, groups=groups))
    train = set(groups[train_pos])
    summary = (
        df.assign(Group_ID=df["Source_dataset"] + "::" + df["Flight_ID"])
        .groupby("Group_ID", as_index=False)
        .agg(Source_dataset=("Source_dataset", "first"), Flight_ID=("Flight_ID", "first"), rows=("Flight_ID", "size"))
    )
    summary["split"] = np.where(summary["Group_ID"].isin(train), "train", "test")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic_smoke.csv")
    parser.add_argument("--manifest", default="data/synthetic_split_manifest.csv")
    parser.add_argument("--n-flights", type=int, default=10)
    parser.add_argument("--rows-per-flight", type=int, default=80)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_flights < 6:
        raise ValueError("Use at least 6 flights so train/validation/test all contain flights.")
    if args.rows_per_flight < 20:
        raise ValueError("Use at least 20 rows per flight for sequence smoke tests.")
    output = Path(args.output)
    manifest = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.n_flights, args.rows_per_flight, args.random_seed)
    df.to_csv(output, index=False)
    write_manifest(df, manifest, args.test_size, args.random_seed)
    print(f"Saved synthetic smoke-test data: {output.resolve()} ({len(df):,} rows)")
    print(f"Saved split manifest: {manifest.resolve()}")
    print("Synthetic results validate software only; they are not research findings.")


if __name__ == "__main__":
    main()
