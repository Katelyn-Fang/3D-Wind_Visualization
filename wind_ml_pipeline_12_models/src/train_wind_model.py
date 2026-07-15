#!/usr/bin/env python3
"""Train one of twelve comparable wind-speed/direction models on a frozen flight split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.neural_models import NEURAL_MODELS, train_neural
from src.tabular_models import TABULAR_MODELS, train_tabular
from src.wind_core import add_engineered_features, load_split, load_standardized_data

SCRIPT_VERSION = "0.3.0"
MODEL_CHOICES = TABULAR_MODELS + NEURAL_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--direction-model",
        choices=TABULAR_MODELS,
        default=None,
        help="Optional separate tabular model family for sine/cosine direction prediction.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--direction-min-speed", type=float, default=1.00)
    parser.add_argument(
        "--direction-target",
        choices=["absolute", "relative_yaw"],
        default="absolute",
        help="Predict absolute direction or direction relative to a verified yaw heading.",
    )
    parser.add_argument(
        "--yaw-transform",
        choices=[
            "clockwise_from_north",
            "counterclockwise_from_north",
            "ccw_from_east_to_heading",
            "cw_from_east_to_heading",
        ],
        default=None,
        help="Required only for --direction-target relative_yaw after convention validation.",
    )
    parser.add_argument(
        "--attitude-angle-unit", choices=["auto", "radians", "degrees"], default="auto",
        help="Unit used by Roll/Pitch/Yaw columns; auto infers from yaw magnitude.",
    )
    parser.add_argument(
        "--comparison-sequence-length",
        type=int,
        default=30,
        help="Defines the common endpoint subset used for fair tabular-vs-sequence metrics.",
    )

    # Tree, boosting, and linear settings.
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=20, help="Use 0 for unlimited depth.")
    parser.add_argument("--max-features", type=float, default=0.80)
    parser.add_argument("--subsample", type=float, default=0.90)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)

    # Neural settings; learning-rate is also used by boosted trees.
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--direction-loss-weight", type=float, default=2.0)
    parser.add_argument("--direction-norm-weight", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # Shared importance interface. --permutation-importance remains accepted for compatibility.
    parser.add_argument("--feature-importance", action="store_true")
    parser.add_argument("--permutation-importance", action="store_true")
    parser.add_argument("--importance-sample-size", type=int, default=1000)
    parser.add_argument("--importance-repeats", type=int, default=2)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1; do not use -1 on the cluster.")
    if args.n_estimators < 1 or args.min_samples_leaf < 1:
        raise ValueError("--n-estimators and --min-samples-leaf must be positive.")
    if not 0.0 < args.max_features <= 1.0:
        raise ValueError("--max-features must be in (0, 1].")
    if not 0.0 < args.subsample <= 1.0:
        raise ValueError("--subsample must be in (0, 1].")
    if args.direction_min_speed < 0:
        raise ValueError("--direction-min-speed cannot be negative.")
    if args.direction_loss_weight < 0 or args.direction_norm_weight < 0:
        raise ValueError("Direction loss weights cannot be negative.")
    if args.direction_model is not None and args.model not in TABULAR_MODELS:
        raise ValueError("--direction-model is supported only when --model is tabular.")
    if args.direction_target == "relative_yaw" and args.yaw_transform is None:
        raise ValueError("--direction-target relative_yaw requires --yaw-transform.")
    if args.sequence_length < 2 and args.model in {"lstm", "tcn", "transformer"}:
        raise ValueError("Sequence models require --sequence-length of at least 2.")
    if args.comparison_sequence_length < 2:
        raise ValueError("--comparison-sequence-length must be at least 2.")
    if args.hidden_size < 4 or args.num_layers < 1 or args.attention_heads < 1:
        raise ValueError("Neural hidden size/layer/head values are too small.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be in (0, 0.5).")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")


def main() -> None:
    args = build_parser().parse_args()
    args.script_version = SCRIPT_VERSION
    args.feature_importance = bool(args.feature_importance or args.permutation_importance)
    validate_args(args)

    data_path = Path(args.data)
    split_path = Path(args.split_manifest)
    output_dir = Path(args.output_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path.resolve()}")
    if not split_path.exists():
        raise FileNotFoundError(f"Could not find split manifest: {split_path.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Model: {args.model}")
    if args.direction_model:
        print(f"Direction model: {args.direction_model}")
    print(f"Direction target: {args.direction_target}")
    df = load_standardized_data(data_path)
    engineered, feature_columns, feature_metadata = add_engineered_features(
        df, attitude_angle_unit=args.attitude_angle_unit
    )
    train_idx, test_idx, _ = load_split(engineered, split_path)
    print(
        f"Rows: {len(engineered):,} total; {len(train_idx):,} train; {len(test_idx):,} test"
    )
    print(
        f"Flights: {engineered.iloc[train_idx]['_Group_ID'].nunique()} train; "
        f"{engineered.iloc[test_idx]['_Group_ID'].nunique()} test"
    )

    if args.model in TABULAR_MODELS:
        metrics = train_tabular(
            args,
            engineered,
            feature_columns,
            feature_metadata,
            train_idx,
            test_idx,
            output_dir,
        )
    else:
        metrics = train_neural(
            args,
            engineered,
            feature_columns,
            feature_metadata,
            train_idx,
            test_idx,
            output_dir,
        )
    print(json.dumps(metrics, indent=2, default=str))
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted.", file=sys.stderr)
        raise SystemExit(130)
