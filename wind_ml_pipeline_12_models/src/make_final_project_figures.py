from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from final_analysis_utils import (
    common_frames,
    compute_common_comparison,
    load_all_predictions,
    select_winners,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the final BU RISE wind-model project figures.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/scc/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/scc/final_analysis"))
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--direction-min-speed", type=float, default=1.0)
    parser.add_argument("--expected-models", type=int, default=12)
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated formats, e.g. png,pdf.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--scatter-max-points", type=int, default=15000)
    parser.add_argument("--confidence-max-points", type=int, default=30000)
    parser.add_argument(
        "--overlay-max-points",
        type=int,
        default=1500,
        help="Maximum evenly spaced rows shown in each multi-model flight overlay.",
    )
    parser.add_argument("--min-flight-rows", type=int, default=100)
    parser.add_argument(
        "--flight-id",
        default=None,
        help="Optional exact internal group ID or Flight_ID suffix for the time-series figure.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def save_figure(fig: plt.Figure, stem: Path, formats: list[str], dpi: int, manifest: list[dict[str, str]]) -> None:
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        manifest.append({"figure": stem.name, "format": fmt, "path": str(path)})
    plt.close(fig)


def annotate_horizontal_bars(ax: plt.Axes, values: np.ndarray, decimals: int = 3) -> None:
    finite = values[np.isfinite(values)]
    span = finite.max() - finite.min() if len(finite) else 1.0
    offset = max(span * 0.015, finite.max() * 0.008 if len(finite) else 0.01)
    for index, value in enumerate(values):
        ax.text(value + offset, index, f"{value:.{decimals}f}", va="center", fontsize=8)


def plot_speed_mae_bar(comparison: pd.DataFrame, figures_dir: Path, formats: list[str], dpi: int, manifest: list[dict[str, str]]) -> None:
    data = comparison.sort_values("common_speed_mae_mps", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    values = data["common_speed_mae_mps"].to_numpy()
    ax.barh(data["display_name"], values)
    ax.invert_yaxis()
    ax.set_xlabel("Common-row wind-speed MAE (m/s)")
    ax.set_title("Wind-Speed MAE by Model")
    annotate_horizontal_bars(ax, values, decimals=3)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figures_dir / "01_speed_mae_by_model", formats, dpi, manifest)


def plot_direction_mae_bar(comparison: pd.DataFrame, threshold: float, figures_dir: Path, formats: list[str], dpi: int, manifest: list[dict[str, str]]) -> None:
    metric = "common_direction_mae_deg_above_threshold"
    data = comparison.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    values = data[metric].to_numpy()
    ax.barh(data["display_name"], values)
    ax.invert_yaxis()
    ax.set_xlabel("Circular direction MAE (degrees)")
    ax.set_title(f"Direction MAE by Model for True Wind Speed > {threshold:g} m/s")
    annotate_horizontal_bars(ax, values, decimals=1)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figures_dir / "02_direction_mae_above_1_mps_by_model", formats, dpi, manifest)


def sample_frame(frame: pd.DataFrame, max_points: int, random_seed: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.sample(max_points, random_state=random_seed)


def plot_speed_scatter(
    frames: dict[str, pd.DataFrame],
    selections: pd.DataFrame,
    figures_dir: Path,
    formats: list[str],
    dpi: int,
    max_points: int,
    random_seed: int,
    manifest: list[dict[str, str]],
) -> None:
    row = selections[selections["category"] == "best_common_speed"].iloc[0]
    model = row["model"]
    frame = sample_frame(frames[model].dropna(subset=["true_speed", "pred_speed"]), max_points, random_seed)
    lower = float(min(frame["true_speed"].min(), frame["pred_speed"].min()))
    upper = float(max(frame["true_speed"].quantile(0.995), frame["pred_speed"].quantile(0.995)))
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    ax.scatter(frame["true_speed"], frame["pred_speed"], s=8, alpha=0.18, rasterized=True)
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("True wind speed (m/s)")
    ax.set_ylabel("Predicted wind speed (m/s)")
    ax.set_title(f"Predicted vs. True Wind Speed — {row['display_name']}")
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, figures_dir / f"03_predicted_vs_true_speed_{model}", formats, dpi, manifest)


def plot_direction_error_distribution(
    frames: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    threshold: float,
    figures_dir: Path,
    formats: list[str],
    dpi: int,
    manifest: list[dict[str, str]],
) -> None:
    order = comparison.sort_values("common_direction_mae_deg_above_threshold")["model"].tolist()
    arrays = []
    labels = []
    for model in order:
        frame = frames[model]
        values = frame.loc[frame["true_speed"] > threshold, "abs_direction_error"].dropna().to_numpy()
        arrays.append(values)
        labels.append(comparison.set_index("model").loc[model, "display_name"])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.boxplot(arrays, vert=False, tick_labels=labels, whis=(5, 95), showfliers=False)
    ax.invert_yaxis()
    ax.set_xlabel("Absolute circular direction error (degrees)")
    ax.set_title(f"Direction-Error Distribution by Model, True Wind Speed > {threshold:g} m/s")
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, figures_dir / "04_direction_error_distribution_by_model", formats, dpi, manifest)


def wind_speed_bins() -> tuple[list[float], list[str]]:
    edges = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, np.inf]
    labels = ["0–0.5", "0.5–1", "1–2", "2–4", "4–6", "6–8", "8–12", "12+"]
    return edges, labels


def plot_direction_mae_by_speed_bin(
    frames: dict[str, pd.DataFrame],
    finalists: pd.DataFrame,
    figures_dir: Path,
    formats: list[str],
    dpi: int,
    manifest: list[dict[str, str]],
) -> None:
    edges, labels = wind_speed_bins()
    fig, ax = plt.subplots(figsize=(9.5, 6))
    table_rows = []
    for row in finalists.itertuples(index=False):
        frame = frames[row.model].copy()
        frame["speed_bin"] = pd.cut(frame["true_speed"], bins=edges, labels=labels, right=False, include_lowest=True)
        grouped = frame.groupby("speed_bin", observed=False)["abs_direction_error"].agg(["mean", "count"]).reindex(labels)
        y = grouped["mean"].where(grouped["count"] >= 30)
        ax.plot(np.arange(len(labels)), y, marker="o", linewidth=1.5, label=row.display_name)
        for label, values in grouped.iterrows():
            table_rows.append({
                "model": row.model,
                "display_name": row.display_name,
                "speed_bin_mps": label,
                "direction_mae_deg": values["mean"],
                "n": int(values["count"]),
            })

    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlabel("True wind-speed bin (m/s)")
    ax.set_ylabel("Circular direction MAE (degrees)")
    ax.set_title("Direction MAE by Wind-Speed Bin — Finalists")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / "05_direction_mae_by_wind_speed_bin", formats, dpi, manifest)
    pd.DataFrame(table_rows).to_csv(figures_dir.parent / "direction_mae_by_wind_speed_bin.csv", index=False)


def per_flight_metrics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    speed = frame.groupby("group_id").agg(
        n=("row_key", "size"),
        speed_mae_mps=("abs_speed_error", "mean"),
    )
    direction = (
        frame[frame["true_speed"] > threshold]
        .groupby("group_id")
        .agg(
            direction_n=("row_key", "size"),
            direction_mae_deg=("abs_direction_error", "mean"),
        )
    )
    return speed.join(direction, how="left").reset_index()


def plot_per_flight_finalists(
    frames: dict[str, pd.DataFrame],
    finalists: pd.DataFrame,
    threshold: float,
    figures_dir: Path,
    formats: list[str],
    dpi: int,
    manifest: list[dict[str, str]],
) -> None:
    labels = finalists["display_name"].tolist()
    speed_arrays = []
    direction_arrays = []
    rows = []
    for row in finalists.itertuples(index=False):
        metrics = per_flight_metrics(frames[row.model], threshold)
        speed_arrays.append(metrics["speed_mae_mps"].dropna().to_numpy())
        direction_arrays.append(metrics["direction_mae_deg"].dropna().to_numpy())
        metrics.insert(0, "model", row.model)
        metrics.insert(1, "display_name", row.display_name)
        rows.append(metrics)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].boxplot(speed_arrays, tick_labels=labels, whis=(5, 95), showfliers=False)
    axes[0].set_ylabel("Per-flight speed MAE (m/s)")
    axes[0].set_title("Speed Error Across Test Flights")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].boxplot(direction_arrays, tick_labels=labels, whis=(5, 95), showfliers=False)
    axes[1].set_ylabel("Per-flight direction MAE (degrees)")
    axes[1].set_title(f"Direction Error Across Test Flights (> {threshold:g} m/s)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Per-Flight Error Comparison for Finalists")
    save_figure(fig, figures_dir / "06_per_flight_error_comparison_finalists", formats, dpi, manifest)
    pd.concat(rows, ignore_index=True).to_csv(figures_dir.parent / "per_flight_metrics_finalists.csv", index=False)


def choose_representative_flight(frame: pd.DataFrame, threshold: float, min_rows: int, requested: str | None) -> str:
    groups = frame["group_id"].astype(str)
    if requested is not None:
        exact = sorted(set(groups[groups == requested]))
        if exact:
            return exact[0]
        suffix = sorted(set(groups[groups.str.endswith(f"::{requested}", na=False)]))
        if len(suffix) == 1:
            return suffix[0]
        raise ValueError(f"Requested flight '{requested}' was not found unambiguously.")

    metrics = per_flight_metrics(frame, threshold)
    metrics = metrics[(metrics["n"] >= min_rows) & metrics["direction_mae_deg"].notna()].copy()
    if metrics.empty:
        metrics = per_flight_metrics(frame, threshold).dropna(subset=["direction_mae_deg"]).copy()
    speed_median = metrics["speed_mae_mps"].median()
    direction_median = metrics["direction_mae_deg"].median()
    speed_scale = max(speed_median, 1e-9)
    direction_scale = max(direction_median, 1e-9)
    metrics["representative_distance"] = (
        (metrics["speed_mae_mps"] - speed_median).abs() / speed_scale
        + (metrics["direction_mae_deg"] - direction_median).abs() / direction_scale
    )
    return str(metrics.sort_values(["representative_distance", "n"], ascending=[True, False]).iloc[0]["group_id"])


def time_axis(flight: pd.DataFrame) -> tuple[np.ndarray, str]:
    elapsed = pd.to_numeric(flight["elapsed_s"], errors="coerce")
    if elapsed.notna().sum() >= 2 and elapsed.nunique(dropna=True) >= 2:
        return elapsed.to_numpy(dtype=float), "Elapsed time (s)"
    timestamp = pd.to_datetime(flight["timestamp"], errors="coerce")
    if timestamp.notna().sum() >= 2 and timestamp.nunique(dropna=True) >= 2:
        relative = (timestamp - timestamp.min()).dt.total_seconds()
        return relative.to_numpy(dtype=float), "Time since flight segment start (s)"
    return pd.to_numeric(flight["sample_index"], errors="coerce").to_numpy(dtype=float), "Sample index"


def plot_representative_flight(
    frames: dict[str, pd.DataFrame],
    selections: pd.DataFrame,
    threshold: float,
    min_rows: int,
    requested_flight: str | None,
    figures_dir: Path,
    formats: list[str],
    dpi: int,
    manifest: list[dict[str, str]],
) -> None:
    row = selections[selections["category"] == "best_balanced_hybrid"].iloc[0]
    model = row["model"]
    frame = frames[model]
    group_id = choose_representative_flight(frame, threshold, min_rows, requested_flight)
    flight = frame[frame["group_id"].astype(str) == group_id].sort_values("sample_index").copy()
    x, x_label = time_axis(flight)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(x, flight["true_speed"], linewidth=1.5, label="True")
    axes[0].plot(x, flight["pred_speed"], linewidth=1.2, label="Predicted")
    axes[0].set_ylabel("Wind speed (m/s)")
    axes[0].set_title(f"Representative Flight: True vs. Predicted Wind — {row['display_name']}")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, flight["true_angle"], linewidth=1.2, label="True")
    axes[1].plot(x, flight["pred_angle"], linewidth=1.0, label="Predicted")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Wind direction (degrees)")
    axes[1].set_ylim(0, 360)
    axes[1].set_yticks([0, 90, 180, 270, 360])
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.text(0.01, 0.01, f"Flight group: {group_id}", fontsize=8)
    save_figure(fig, figures_dir / f"07_representative_flight_time_series_{model}", formats, dpi, manifest)
    (figures_dir.parent / "representative_flight.txt").write_text(
        f"model={model}\ndisplay_name={row['display_name']}\ngroup_id={group_id}\nrows={len(flight)}\n",
        encoding="utf-8",
    )


def plot_confidence_vs_error(
    frames: dict[str, pd.DataFrame],
    selections: pd.DataFrame,
    threshold: float,
    figures_dir: Path,
    formats: list[str],
    dpi: int,
    max_points: int,
    random_seed: int,
    manifest: list[dict[str, str]],
) -> None:
    row = selections[selections["category"] == "best_balanced_hybrid"].iloc[0]
    model = row["model"]
    full = frames[model]
    full = full[
        (full["true_speed"] > threshold)
        & full["confidence"].notna()
        & full["abs_direction_error"].notna()
    ].copy()
    if full.empty:
        note = figures_dir / "08_direction_confidence_vs_error_SKIPPED.txt"
        note.write_text(
            f"{row['display_name']} has no finite Predicted_direction_confidence values.",
            encoding="utf-8",
        )
        manifest.append({"figure": note.stem, "format": "txt", "path": str(note)})
        return

    sample = sample_frame(full, max_points, random_seed)
    quantiles = np.linspace(0.0, 1.0, 11)
    edges = np.unique(full["confidence"].quantile(quantiles).to_numpy())
    if len(edges) >= 3:
        full["confidence_bin"] = pd.cut(full["confidence"], bins=edges, include_lowest=True, duplicates="drop")
        binned = full.groupby("confidence_bin", observed=True).agg(
            confidence=("confidence", "median"),
            median_error=("abs_direction_error", "median"),
            q25_error=("abs_direction_error", lambda x: x.quantile(0.25)),
            q75_error=("abs_direction_error", lambda x: x.quantile(0.75)),
            n=("abs_direction_error", "size"),
        ).reset_index(drop=True)
    else:
        binned = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(sample["confidence"], sample["abs_direction_error"], s=7, alpha=0.12, rasterized=True, label="Prediction rows")
    if not binned.empty:
        ax.plot(binned["confidence"], binned["median_error"], marker="o", linewidth=1.8, label="Binned median error")
        ax.fill_between(
            binned["confidence"].to_numpy(dtype=float),
            binned["q25_error"].to_numpy(dtype=float),
            binned["q75_error"].to_numpy(dtype=float),
            alpha=0.15,
            label="Binned interquartile range",
        )
    ax.set_xlabel("Predicted direction confidence")
    ax.set_ylabel("Absolute circular direction error (degrees)")
    ax.set_title(f"Direction Confidence vs. Direction Error — {row['display_name']}")
    ax.set_ylim(0, 180)
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures_dir / f"08_direction_confidence_vs_error_{model}", formats, dpi, manifest)
    if not binned.empty:
        binned.to_csv(figures_dir.parent / "direction_confidence_error_bins.csv", index=False)


def select_three_unique_finalists(
    comparison: pd.DataFrame,
    selections: pd.DataFrame,
) -> list[str]:
    """
    Return exactly three unique finalist models when at least three models exist.

    Priority is:
      1. best common speed model;
      2. best direction model above the threshold;
      3. best balanced model.

    If one model wins multiple categories, remaining slots are filled using the
    balanced ranking so the overlay still contains three distinct models.
    """
    category_order = [
        "best_common_speed",
        "best_direction_above_threshold",
        "best_balanced_hybrid",
    ]

    selected: list[str] = []
    for category in category_order:
        matches = selections.loc[selections["category"] == category, "model"]
        if not matches.empty:
            model = str(matches.iloc[0])
            if model not in selected:
                selected.append(model)

    ranked_models = (
        comparison.sort_values(
            [
                "balanced_rank",
                "balanced_hybrid_score",
                "common_speed_mae_mps",
                "common_direction_mae_deg_above_threshold",
                "model",
            ]
        )["model"]
        .astype(str)
        .tolist()
    )
    for model in ranked_models:
        if model not in selected:
            selected.append(model)
        if len(selected) == 3:
            break

    return selected[:3]


def align_flight_predictions(
    frames: dict[str, pd.DataFrame],
    model_names: list[str],
    group_id: str,
) -> pd.DataFrame:
    """Align observed values and selected model predictions by exact row_key."""
    if not model_names:
        raise ValueError("At least one model is required for an overlay.")

    reference_model = model_names[0]
    reference = frames[reference_model]
    aligned = reference.loc[
        reference["group_id"].astype(str) == str(group_id),
        [
            "row_key",
            "sample_index",
            "timestamp",
            "elapsed_s",
            "true_speed",
            "true_angle",
        ],
    ].copy()

    if aligned.empty:
        raise ValueError(
            f"No common prediction rows were found for flight group '{group_id}'."
        )

    for model in model_names:
        prediction_rows = frames[model].loc[
            frames[model]["group_id"].astype(str) == str(group_id),
            ["row_key", "pred_speed", "pred_angle"],
        ].copy()
        prediction_rows = prediction_rows.rename(
            columns={
                "pred_speed": f"pred_speed__{model}",
                "pred_angle": f"pred_angle__{model}",
            }
        )
        aligned = aligned.merge(
            prediction_rows,
            on="row_key",
            how="inner",
            validate="one_to_one",
        )

    if aligned.empty:
        raise ValueError(
            f"The selected models have no aligned rows for flight group '{group_id}'."
        )

    return aligned.sort_values(["sample_index", "row_key"]).reset_index(drop=True)


def evenly_downsample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Keep evenly spaced sequence rows without changing their order."""
    if max_points <= 0:
        raise ValueError("overlay-max-points must be positive.")
    if len(frame) <= max_points:
        return frame

    positions = np.unique(
        np.rint(np.linspace(0, len(frame) - 1, max_points)).astype(int)
    )
    return frame.iloc[positions].reset_index(drop=True)


def overlay_time_axis(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Choose elapsed time, timestamp-relative time, or sample index."""
    elapsed = pd.to_numeric(frame["elapsed_s"], errors="coerce")
    if elapsed.notna().sum() >= 2 and elapsed.nunique(dropna=True) >= 2:
        return elapsed.to_numpy(dtype=float), "Elapsed time (s)"

    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamp.notna().sum() >= 2 and timestamp.nunique(dropna=True) >= 2:
        relative = (timestamp - timestamp.min()).dt.total_seconds()
        return relative.to_numpy(dtype=float), "Time since flight segment start (s)"

    sample_index = pd.to_numeric(frame["sample_index"], errors="coerce")
    return sample_index.to_numpy(dtype=float), "Within-flight sample index"


def plot_multi_model_flight_overlay(
    frames: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    model_names: list[str],
    group_id: str,
    title: str,
    output_stem: str,
    figures_dir: Path,
    formats: list[str],
    dpi: int,
    max_points: int,
    manifest: list[dict[str, str]],
) -> None:
    """
    Plot observed and predicted wind speed/direction for one shared flight.

    Both panels use the same aligned common rows. The observed series is a
    thicker black line, while model predictions use thinner lines.
    """
    aligned_full = align_flight_predictions(frames, model_names, group_id)
    aligned = evenly_downsample(aligned_full, max_points)
    x, x_label = overlay_time_axis(aligned)

    display_names = comparison.set_index("model")["display_name"].to_dict()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
    )

    observed_speed_line, = axes[0].plot(
        x,
        aligned["true_speed"],
        color="black",
        linewidth=3.0,
        label="Observed",
        zorder=20,
    )
    axes[1].plot(
        x,
        aligned["true_angle"],
        color="black",
        linewidth=3.0,
        label="Observed",
        zorder=20,
    )

    model_lines = []
    for model in model_names:
        label = str(display_names.get(model, model.replace("_", " ").title()))
        speed_line, = axes[0].plot(
            x,
            aligned[f"pred_speed__{model}"],
            linewidth=1.2,
            alpha=0.78,
            label=label,
        )
        axes[1].plot(
            x,
            aligned[f"pred_angle__{model}"],
            linewidth=1.2,
            alpha=0.78,
            label=label,
        )
        model_lines.append(speed_line)

    axes[0].set_ylabel("Wind speed (m/s)")
    axes[0].set_title("Observed and Predicted Wind Speed")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Wind direction (degrees)")
    axes[1].set_title("Observed and Predicted Wind Direction")
    axes[1].set_ylim(0, 360)
    axes[1].set_yticks([0, 90, 180, 270, 360])
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        f"{title}\n"
        f"Flight {group_id}; {len(aligned):,} plotted rows "
        f"from {len(aligned_full):,} aligned rows",
        y=0.985,
    )

    legend_handles = [observed_speed_line, *model_lines]
    legend_labels = ["Observed", *[
        str(display_names.get(model, model.replace("_", " ").title()))
        for model in model_names
    ]]
    legend_columns = 4 if len(legend_handles) > 4 else len(legend_handles)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=legend_columns,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save_figure(
        fig,
        figures_dir / output_stem,
        formats,
        dpi,
        manifest,
    )


def plot_final_model_overlays(
    frames: dict[str, pd.DataFrame],
    comparison: pd.DataFrame,
    selections: pd.DataFrame,
    threshold: float,
    min_rows: int,
    requested_flight: str | None,
    figures_dir: Path,
    formats: list[str],
    dpi: int,
    max_points: int,
    manifest: list[dict[str, str]],
) -> None:
    """Create the all-model and three-finalist speed/direction overlays."""
    balanced_row = selections[
        selections["category"] == "best_balanced_hybrid"
    ].iloc[0]
    balanced_model = str(balanced_row["model"])

    group_id = choose_representative_flight(
        frames[balanced_model],
        threshold,
        min_rows,
        requested_flight,
    )

    all_models = (
        comparison.sort_values(
            [
                "balanced_rank",
                "speed_rank",
                "direction_rank",
                "model",
            ]
        )["model"]
        .astype(str)
        .tolist()
    )
    top_three = select_three_unique_finalists(comparison, selections)

    plot_multi_model_flight_overlay(
        frames=frames,
        comparison=comparison,
        model_names=all_models,
        group_id=group_id,
        title="All Models Compared with Observed Wind",
        output_stem="09_all_models_wind_speed_and_direction_overlay",
        figures_dir=figures_dir,
        formats=formats,
        dpi=dpi,
        max_points=max_points,
        manifest=manifest,
    )

    plot_multi_model_flight_overlay(
        frames=frames,
        comparison=comparison,
        model_names=top_three,
        group_id=group_id,
        title="Three Final Models Compared with Observed Wind",
        output_stem="10_top_3_models_wind_speed_and_direction_overlay",
        figures_dir=figures_dir,
        formats=formats,
        dpi=dpi,
        max_points=max_points,
        manifest=manifest,
    )

    finalist_names = comparison.set_index("model")["display_name"].to_dict()
    summary_lines = [
        f"flight_group={group_id}",
        f"all_model_count={len(all_models)}",
        "top_three_models=" + ",".join(top_three),
        "top_three_display_names="
        + ",".join(str(finalist_names.get(model, model)) for model in top_three),
        f"overlay_max_points={max_points}",
    ]
    (figures_dir.parent / "model_overlay_details.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    formats = [value.strip().lower() for value in args.formats.split(",") if value.strip()]
    unsupported = [value for value in formats if value not in {"png", "pdf", "svg"}]
    if unsupported:
        raise ValueError(f"Unsupported figure formats: {unsupported}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.figures_dir or (args.output_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    models = load_all_predictions(args.results_dir, expected_models=args.expected_models)
    comparison, keys = compute_common_comparison(
        models,
        direction_min_speed=args.direction_min_speed,
        speed_weight=0.5,
        direction_weight=0.5,
    )
    selections, finalists = select_winners(comparison)
    frames = common_frames(models, keys)

    comparison.to_csv(args.output_dir / "model_comparison_common_recomputed.csv", index=False)
    selections.to_csv(args.output_dir / "final_model_selection.csv", index=False)
    finalists.to_csv(args.output_dir / "finalists_unique.csv", index=False)

    manifest: list[dict[str, str]] = []
    plot_speed_mae_bar(comparison, figures_dir, formats, args.dpi, manifest)
    plot_direction_mae_bar(comparison, args.direction_min_speed, figures_dir, formats, args.dpi, manifest)
    plot_speed_scatter(frames, selections, figures_dir, formats, args.dpi, args.scatter_max_points, args.random_seed, manifest)
    plot_direction_error_distribution(frames, comparison, args.direction_min_speed, figures_dir, formats, args.dpi, manifest)
    plot_direction_mae_by_speed_bin(frames, finalists, figures_dir, formats, args.dpi, manifest)
    plot_per_flight_finalists(frames, finalists, args.direction_min_speed, figures_dir, formats, args.dpi, manifest)
    plot_representative_flight(
        frames,
        selections,
        args.direction_min_speed,
        args.min_flight_rows,
        args.flight_id,
        figures_dir,
        formats,
        args.dpi,
        manifest,
    )
    plot_confidence_vs_error(
        frames,
        selections,
        args.direction_min_speed,
        figures_dir,
        formats,
        args.dpi,
        args.confidence_max_points,
        args.random_seed,
        manifest,
    )
    plot_final_model_overlays(
        frames,
        comparison,
        selections,
        args.direction_min_speed,
        args.min_flight_rows,
        args.flight_id,
        figures_dir,
        formats,
        args.dpi,
        args.overlay_max_points,
        manifest,
    )

    manifest_path = args.output_dir / "figure_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"Created {len(manifest)} figure files under {figures_dir}")
    print(f"Figure manifest: {manifest_path}")


if __name__ == "__main__":
    main()
