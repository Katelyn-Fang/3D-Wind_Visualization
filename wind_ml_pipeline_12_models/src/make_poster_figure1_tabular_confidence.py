#!/usr/bin/env python3
"""Regenerate poster Figure 1 with explicit, consistent evaluation scopes."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "scc" / "baseline" / "cpu"
OUTPUT = ROOT / "results" / "scc" / "final_analysis" / "poster"
THRESHOLD = 1.0
SEED = 42
N_BOOTSTRAP = 5000

DISPLAY_NAMES = {
    "dummy": "Dummy",
    "ridge": "Ridge",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def load_predictions() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(BASELINE.glob("*/test_predictions.csv")):
        frame = pd.read_csv(path)
        model = str(frame["Speed_model_name"].iloc[0]).lower()
        if model not in DISPLAY_NAMES:
            continue
        frame = frame.rename(
            columns={
                "_Group_ID": "group_id",
                "Wind_speed": "true_speed",
                "Speed_error_mps": "speed_error",
                "Angle_error_deg": "direction_error",
                "Predicted_direction_confidence": "confidence",
            }
        )
        frame["abs_speed_error"] = frame["speed_error"].abs()
        frame["abs_direction_error"] = frame["direction_error"].abs()
        frames[model] = frame
    missing = set(DISPLAY_NAMES) - set(frames)
    if missing:
        raise FileNotFoundError(f"Missing tabular predictions: {sorted(missing)}")
    return frames


def summarize(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    # A flight is eligible only if every model has speed predictions and at least
    # one direction observation above the stated threshold. This reproduces the
    # 52-flight scope used by the poster's flight-balanced comparison.
    common: set[str] | None = None
    for frame in frames.values():
        frame = frame.loc[frame["Common_sequence_eligible"].astype(bool)]
        direction_groups = set(
            frame.loc[frame["true_speed"] > THRESHOLD, "group_id"].astype(str)
        )
        speed_groups = set(frame["group_id"].astype(str))
        valid = direction_groups & speed_groups
        common = valid if common is None else common & valid
    common_flights = sorted(common or [])
    if not common_flights:
        raise RuntimeError("No common direction-eligible flights found")

    rows = []
    for index, (model, frame) in enumerate(sorted(frames.items())):
        confidence_rows = frame.copy()
        accuracy_rows = frame[
            frame["group_id"].astype(str).isin(common_flights)
            & frame["Common_sequence_eligible"].astype(bool)
        ].copy()
        speed_by_flight = accuracy_rows.groupby("group_id")["abs_speed_error"].mean()
        direction_rows = accuracy_rows[accuracy_rows["true_speed"] > THRESHOLD].copy()
        direction_by_flight = direction_rows.groupby("group_id")[
            "abs_direction_error"
        ].mean()
        # Confidence diagnostics retain the original Figure 1 all-row scope so
        # the reported Spearman statistic remains directly reproducible.
        confidence_by_flight = confidence_rows.groupby("group_id")["confidence"].mean()

        speed_low, speed_high = bootstrap_ci(speed_by_flight.to_numpy(), SEED + index)
        direction_low, direction_high = bootstrap_ci(
            direction_by_flight.to_numpy(), SEED + 100 + index
        )
        spearman = confidence_rows[["confidence", "abs_direction_error"]].corr(
            method="spearman"
        ).iloc[0, 1]
        rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "n_flights": len(common_flights),
                "speed_mae_mps": speed_by_flight.mean(),
                "speed_ci_low": speed_low,
                "speed_ci_high": speed_high,
                "direction_mae_deg": direction_by_flight.mean(),
                "direction_ci_low": direction_low,
                "direction_ci_high": direction_high,
                "mean_direction_confidence": confidence_by_flight.mean(),
                "confidence_error_spearman": spearman,
            }
        )
    return pd.DataFrame(rows), common_flights


def add_value_labels(ax: plt.Axes, bars, fmt: str) -> None:
    values = [bar.get_width() for bar in bars]
    span = max(values) - min(0, min(values))
    pad = max(span * 0.018, 0.008)
    for bar, value in zip(bars, values):
        x = value + pad
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(value),
            va="center",
            ha="left",
            fontsize=8.5,
        )


def plot(summary: pd.DataFrame, n_flights: int) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))

    panels = [
        ("speed_mae_mps", "Flight-balanced wind-speed error", "MAE (m/s)", "#2878B5", True, "{:.3f}"),
        ("direction_mae_deg", "Flight-balanced wind-direction error", "Circular MAE (degrees)", "#D95319", True, "{:.1f}"),
        ("mean_direction_confidence", "Mean direction confidence", "Predicted direction-vector magnitude", "#3A923A", False, "{:.3f}"),
        ("confidence_error_spearman", "Does confidence track direction error?", "Spearman correlation (more negative is better)", "#8E5AA9", True, "{:.2f}"),
    ]

    for ax, (column, title, xlabel, color, ascending, fmt) in zip(axes.flat, panels):
        ordered = summary.dropna(subset=[column]).sort_values(column, ascending=ascending)
        bars = ax.barh(ordered["display_name"], ordered[column], color=color, alpha=0.95)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, weight="bold", pad=21)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
        ax.set_axisbelow(True)
        if column == "confidence_error_spearman":
            ax.axvline(0, color="#333333", linewidth=1)
            ax.set_xlim(min(ordered[column].min() * 1.14, -0.1), 0)
        else:
            ax.set_xlim(0, ordered[column].max() * 1.14)
        add_value_labels(ax, bars, fmt)

    axes[0, 0].text(
        0.0, 1.01, f"All rows; {n_flights} eligible test flights weighted equally",
        transform=axes[0, 0].transAxes, fontsize=8, color="#444444"
    )
    axes[0, 1].text(
        0.0, 1.01, f"True wind speed > {THRESHOLD:g} m/s; {n_flights} flights weighted equally",
        transform=axes[0, 1].transAxes, fontsize=8, color="#444444"
    )
    for ax in axes[1]:
        ax.text(
            0.0, 1.01, "All held-out test rows",
            transform=ax.transAxes, fontsize=8, color="#444444"
        )

    fig.suptitle("Tabular Model Accuracy and Direction Confidence", fontsize=15, weight="bold")
    fig.text(
        0.5,
        0.012,
        "Accuracy panels use the same direction filter and equal-flight weighting as Figure 2. "
        "Confidence is the predicted direction-vector magnitude, not a calibrated probability.",
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    fig.tight_layout(rect=[0.02, 0.045, 0.99, 0.95], h_pad=2.4, w_pad=2.6)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "figure1_tabular_accuracy_confidence_corrected.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure1_tabular_accuracy_confidence_corrected.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "figure1_tabular_accuracy_confidence_expanded_final.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT / "figure1_tabular_accuracy_confidence_expanded_final.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary, common_flights = summarize(load_predictions())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.sort_values("speed_mae_mps").to_csv(
        OUTPUT / "figure1_tabular_accuracy_confidence_corrected.csv", index=False
    )
    plot(summary, len(common_flights))
    extra_trees = summary.set_index("model").loc["extra_trees"]
    print(f"Eligible flights: {len(common_flights)}")
    print(extra_trees.to_string())


if __name__ == "__main__":
    main()
