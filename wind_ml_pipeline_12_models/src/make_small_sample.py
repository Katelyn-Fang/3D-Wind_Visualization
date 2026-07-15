#!/usr/bin/env python3
"""Create a reproducible, flight-preserving development sample and split manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_group_id(df: pd.DataFrame) -> pd.Series:
    source = (
        df["Source_dataset"].astype("string").fillna("unknown")
        if "Source_dataset" in df.columns
        else pd.Series("unknown", index=df.index, dtype="string")
    )
    return source + "::" + df["Flight_ID"].astype("string").fillna("missing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select whole flights for a small development dataset and create one fixed "
            "train/test split shared by every model."
        )
    )
    parser.add_argument("--data", required=True, help="Full standardized telemetry CSV.")
    parser.add_argument("--output", default="data/small_sample.csv")
    parser.add_argument("--manifest", default="data/small_split_manifest.csv")
    parser.add_argument("--n-flights", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--max-rows-per-flight",
        type=int,
        default=0,
        help=(
            "Optional contiguous row cap per flight for a fast smoke test. "
            "Use 0 to retain every row in selected flights."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_flights < 4:
        raise ValueError("Use at least 4 flights so the development split is meaningful.")
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be strictly between 0 and 1.")
    if args.max_rows_per_flight < 0:
        raise ValueError("--max-rows-per-flight cannot be negative.")

    input_path = Path(args.data)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    required = {"Flight_ID", "Wind_speed", "Wind_angle"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "Source_dataset" not in df.columns:
        df["Source_dataset"] = "unknown"

    df = df.dropna(subset=["Flight_ID", "Wind_speed", "Wind_angle"]).copy()
    df["_Group_ID"] = make_group_id(df)
    available_groups = np.array(sorted(df["_Group_ID"].unique()), dtype=object)
    if len(available_groups) < 4:
        raise ValueError(f"Found only {len(available_groups)} usable flights; at least 4 are required.")

    n_select = min(args.n_flights, len(available_groups))
    rng = np.random.default_rng(args.random_seed)
    selected_groups = set(rng.choice(available_groups, size=n_select, replace=False).tolist())
    selected = df.loc[df["_Group_ID"].isin(selected_groups)].copy()

    if args.max_rows_per_flight:
        pieces = []
        for _, part in selected.groupby("_Group_ID", sort=True, observed=True):
            if len(part) <= args.max_rows_per_flight:
                pieces.append(part)
                continue
            max_start = len(part) - args.max_rows_per_flight
            start = int(rng.integers(0, max_start + 1))
            pieces.append(part.iloc[start : start + args.max_rows_per_flight])
        selected = pd.concat(pieces, axis=0).sort_index(kind="stable")

    group_ids = np.array(sorted(selected["_Group_ID"].unique()), dtype=object)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_seed
    )
    # One representative row per group makes the requested fraction apply to flights.
    representatives = pd.DataFrame({"Group_ID": group_ids})
    train_pos, test_pos = next(
        splitter.split(representatives, groups=representatives["Group_ID"])
    )
    train_groups = set(representatives.iloc[train_pos]["Group_ID"])
    test_groups = set(representatives.iloc[test_pos]["Group_ID"])

    summary = (
        selected.groupby("_Group_ID", sort=True, observed=True)
        .agg(
            Source_dataset=("Source_dataset", "first"),
            Flight_ID=("Flight_ID", "first"),
            rows=("_Group_ID", "size"),
        )
        .reset_index()
        .rename(columns={"_Group_ID": "Group_ID"})
    )
    summary["split"] = np.where(summary["Group_ID"].isin(train_groups), "train", "test")
    summary.to_csv(manifest_path, index=False)

    selected = selected.drop(columns=["_Group_ID"])
    selected.to_csv(output_path, index=False)

    print(f"Input rows: {len(df):,}")
    print(f"Selected rows: {len(selected):,}")
    print(f"Selected flights: {len(group_ids)}")
    print(f"Train flights: {len(train_groups)}; test flights: {len(test_groups)}")
    print(f"Saved sample: {output_path.resolve()}")
    print(f"Saved fixed split: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
