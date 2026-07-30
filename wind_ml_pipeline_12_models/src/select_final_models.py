from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from final_analysis_utils import (
    compute_common_comparison,
    load_all_predictions,
    select_winners,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare all active wind-model predictions on exact common rows and select "
            "the speed, direction, and balanced finalists."
        )
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/scc/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/scc/final_analysis"))
    parser.add_argument("--direction-min-speed", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=0.5)
    parser.add_argument("--direction-weight", type=float, default=0.5)
    parser.add_argument(
        "--expected-models",
        type=int,
        default=12,
        help="Expected active model count. Use 0 to disable the count check.",
    )
    return parser


def write_text_report(
    output_path: Path,
    comparison: pd.DataFrame,
    selections: pd.DataFrame,
    finalists: pd.DataFrame,
) -> None:
    lines = [
        "BU RISE wind-model final selection",
        "=" * 38,
        "",
        f"Models compared: {len(comparison)}",
        f"Exact common rows per model: {int(comparison['common_n'].iloc[0]):,}",
        f"Direction threshold: true wind speed > {comparison['direction_speed_threshold_mps'].iloc[0]:g} m/s",
        "",
    ]
    for row in selections.itertuples(index=False):
        lines.extend([
            row.category.replace("_", " ").title(),
            f"  Model: {row.display_name} ({row.model})",
            f"  Value: {row.value:.6f} {row.units}",
            f"  Rule: {row.definition}",
            "",
        ])

    lines.extend([
        "Unique finalist models",
        "-" * 22,
    ])
    for row in finalists.itertuples(index=False):
        lines.append(
            f"  {row.display_name}: speed MAE={row.common_speed_mae_mps:.4f} m/s; "
            f"direction MAE={row.common_direction_mae_deg_above_threshold:.2f} deg; "
            f"balanced score={row.balanced_hybrid_score:.4f}; roles={row.roles}"
        )

    lines.extend([
        "",
        "Balanced-score interpretation",
        "-" * 29,
        "A score of 1.000 would match the best observed speed MAE and the best observed direction MAE.",
        "The score is the weighted geometric mean of each model's MAE relative to the best MAE.",
        "It identifies the strongest joint compromise and does not blend predictions from multiple models.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = load_all_predictions(args.results_dir, expected_models=args.expected_models)
    comparison, _ = compute_common_comparison(
        models,
        direction_min_speed=args.direction_min_speed,
        speed_weight=args.speed_weight,
        direction_weight=args.direction_weight,
    )
    selections, finalists = select_winners(comparison)

    comparison_path = args.output_dir / "model_comparison_common_recomputed.csv"
    selection_path = args.output_dir / "final_model_selection.csv"
    finalists_path = args.output_dir / "finalists_unique.csv"
    report_path = args.output_dir / "final_model_selection.txt"

    comparison.to_csv(comparison_path, index=False)
    selections.to_csv(selection_path, index=False)
    finalists.to_csv(finalists_path, index=False)
    write_text_report(report_path, comparison, selections, finalists)

    print(comparison[
        [
            "display_name",
            "common_speed_mae_mps",
            "common_speed_r",
            "common_direction_mae_deg_above_threshold",
            "balanced_hybrid_score",
            "speed_rank",
            "direction_rank",
            "balanced_rank",
        ]
    ].to_string(index=False))
    print()
    print(selections[["category", "display_name", "value", "units"]].to_string(index=False))
    print(f"\nWrote:\n  {comparison_path}\n  {selection_path}\n  {finalists_path}\n  {report_path}")


if __name__ == "__main__":
    main()
