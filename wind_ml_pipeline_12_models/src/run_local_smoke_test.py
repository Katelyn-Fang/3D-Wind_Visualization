#!/usr/bin/env python3
"""Run all twelve models locally on one small frozen flight split."""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

MODELS = [
    "dummy", "ridge", "decision_tree", "random_forest", "extra_trees",
    "xgboost", "lightgbm", "catboost", "mlp", "lstm", "tcn", "transformer",
]
NEURAL = {"mlp", "lstm", "tcn", "transformer"}


def run_command(command: list[str], log_path: Path) -> tuple[int, float]:
    start = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    duration = time.perf_counter() - start
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + completed.stdout
        + "\n\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    return completed.returncode, duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_smoke.csv")
    parser.add_argument("--split-manifest", default="data/synthetic_split_manifest.csv")
    parser.add_argument("--results-dir", default="results/local_smoke_test")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--direction-min-speed", type=float, default=1.0)
    parser.add_argument("--direction-loss-weight", type=float, default=2.0)
    parser.add_argument("--direction-norm-weight", type=float, default=0.05)
    parser.add_argument("--direction-target", choices=["absolute", "relative_yaw"], default="absolute")
    parser.add_argument("--yaw-transform", default=None)
    parser.add_argument("--attitude-angle-unit", choices=["auto", "radians", "degrees"], default="auto")
    parser.add_argument("--direction-model", choices=MODELS[:8], default=None)
    parser.add_argument("--quick", action="store_true", help="Use the smallest settings that test code paths.")
    parser.add_argument("--feature-importance", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = (root / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    manifest = (
        (root / args.split_manifest).resolve()
        if not Path(args.split_manifest).is_absolute()
        else Path(args.split_manifest)
    )
    results = (
        (root / args.results_dir).resolve()
        if not Path(args.results_dir).is_absolute()
        else Path(args.results_dir)
    )
    if not data.exists() or not manifest.exists():
        command = [
            sys.executable, str(root / "src" / "generate_synthetic_data.py"),
            "--output", str(data), "--manifest", str(manifest),
            "--n-flights", "10", "--rows-per-flight", "64" if args.quick else "100",
            "--random-seed", str(args.random_seed),
        ]
        subprocess.run(command, check=True)
    if results.exists() and not args.keep_existing:
        shutil.rmtree(results)
    results.mkdir(parents=True, exist_ok=True)

    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    status_rows = []
    for number, model in enumerate(requested, start=1):
        out = results / f"{number:02d}_{model}"
        log = results / "logs" / f"{number:02d}_{model}.log"
        common = [
            sys.executable,
            str(root / "src" / "train_wind_model.py"),
            "--data", str(data),
            "--split-manifest", str(manifest),
            "--output-dir", str(out),
            "--model", model,
            "--random-seed", str(args.random_seed),
            "--n-jobs", str(args.n_jobs),
            "--comparison-sequence-length", str(args.sequence_length),
            "--sequence-length", str(args.sequence_length),
            "--direction-min-speed", str(args.direction_min_speed),
            "--direction-loss-weight", str(args.direction_loss_weight),
            "--direction-norm-weight", str(args.direction_norm_weight),
            "--direction-target", args.direction_target,
            "--attitude-angle-unit", args.attitude_angle_unit,
        ]
        if args.yaw_transform:
            common += ["--yaw-transform", args.yaw_transform]
        if args.direction_model and model not in NEURAL:
            common += ["--direction-model", args.direction_model]
        if model in NEURAL:
            common += [
                "--epochs", "1" if args.quick else "5",
                "--batch-size", "64",
                "--hidden-size", "16" if args.quick else "48",
                "--num-layers", "1" if args.quick else "2",
                "--attention-heads", "2" if args.quick else "4",
                "--early-stopping-patience", "2",
                "--device", "cpu",
                "--learning-rate", "0.001",
            ]
        else:
            common += [
                "--n-estimators", "8" if args.quick else "40",
                "--min-samples-leaf", "3",
                "--max-depth", "8",
                "--max-features", "0.8",
                "--learning-rate", "0.05",
            ]
        if args.feature_importance:
            common += ["--feature-importance", "--importance-sample-size", "100", "--importance-repeats", "1"]

        print(f"[{number:02d}/{len(requested):02d}] {model} ...", flush=True)
        code, duration = run_command(common, log)
        validation_error = ""
        if code == 0:
            prediction_path = out / "test_predictions.csv"
            required_columns = {
                "Predicted_direction_confidence",
                "Direction_target_mode",
                "Speed_model_name",
                "Direction_model_name",
            }
            try:
                prediction_columns = set(pd.read_csv(prediction_path, nrows=1).columns)
                missing_columns = sorted(required_columns - prediction_columns)
                if missing_columns:
                    validation_error = f"missing prediction columns: {missing_columns}"
                    code = 2
            except Exception as exc:  # pragma: no cover - surfaced in smoke-test log/status
                validation_error = f"prediction validation failed: {exc}"
                code = 2
        status = "PASS" if code == 0 else "FAIL"
        print(f"    {status} in {duration:.1f}s; log={log}")
        status_rows.append({
            "model": model,
            "status": status,
            "return_code": code,
            "wall_seconds": round(duration, 3),
            "output_directory": str(out),
            "log_file": str(log),
            "validation_error": validation_error,
        })

    status_path = results / "smoke_test_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=status_rows[0].keys())
        writer.writeheader()
        writer.writerows(status_rows)

    successful = [row for row in status_rows if row["status"] == "PASS"]
    if successful:
        subprocess.run(
            [sys.executable, str(root / "src" / "summarize_runs.py"), "--results-dir", str(results)],
            check=True,
        )
    failures = [row for row in status_rows if row["status"] != "PASS"]
    print(f"\nSmoke test: {len(successful)}/{len(status_rows)} models passed.")
    print(f"Status file: {status_path}")
    if failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
