#!/usr/bin/env python3
"""Create corrected poster figures without replacing prior outputs."""

from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scc" / "final_analysis" / "poster_regenerated"
TABULAR_SUMMARY = (
    ROOT / "results" / "scc" / "final_analysis" / "poster"
    / "figure1_tabular_accuracy_confidence_corrected.csv"
)
ALL_MODEL_SUMMARY = (
    ROOT / "results" / "scc" / "final_analysis"
    / "average_errors_all_flights_all_models.csv"
)
PRIMARY_PREDICTIONS = (
    ROOT / "results" / "scc" / "baseline" / "cpu" / "1_extra_trees"
    / "test_predictions.csv"
)
AMOVFLY_PREDICTIONS = (
    ROOT / "results" / "external_amovfly" / "extra_trees"
    / "test_predictions.csv"
)
DRONE_ONBOARD_PREDICTIONS = (
    ROOT / "results" / "external_drone_onboard" / "predictions" / "extra_trees.csv"
)


def circular_residual(predicted, observed) -> np.ndarray:
    return (np.asarray(predicted) - np.asarray(observed) + 180.0) % 360.0 - 180.0


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def label_bars(ax: plt.Axes, bars, fmt: str, *, inside_negative: bool = False) -> None:
    widths = np.array([bar.get_width() for bar in bars], dtype=float)
    pad = max(np.nanmax(np.abs(widths)) * 0.012, 0.006)
    for bar, value in zip(bars, widths):
        if not np.isfinite(value):
            continue
        x = value + pad
        ax.text(x, bar.get_y() + bar.get_height() / 2, fmt.format(value),
                va="center", ha="left", fontsize=8.5)


def figure1_confidence_only() -> None:
    data = pd.read_csv(TABULAR_SUMMARY)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))

    ordered = data.sort_values("mean_direction_confidence", ascending=False)
    bars = axes[0].barh(ordered["display_name"], ordered["mean_direction_confidence"],
                        color="#3A923A")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.06)
    axes[0].set_title("Mean direction confidence", weight="bold")
    axes[0].set_xlabel("Predicted direction-vector magnitude")
    label_bars(axes[0], bars, "{:.3f}")

    ordered = data.dropna(subset=["confidence_error_spearman"]).sort_values(
        "confidence_error_spearman"
    )
    bars = axes[1].barh(ordered["display_name"], ordered["confidence_error_spearman"],
                        color="#8E5AA9")
    axes[1].invert_yaxis()
    axes[1].set_xlim(min(-0.7, ordered["confidence_error_spearman"].min() * 1.08), 0)
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_title("Confidence versus direction error", weight="bold")
    axes[1].set_xlabel("Spearman correlation (more negative is better)")
    label_bars(axes[1], bars, "{:.2f}", inside_negative=True)

    for ax in axes:
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
    fig.suptitle("Tabular Direction-Confidence Diagnostics", fontsize=15, weight="bold")
    fig.text(0.5, 0.015,
             "Both panels use all held-out DJI test rows. Confidence is the predicted sine/cosine vector magnitude, not a calibrated probability.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0.02, 0.055, 0.99, 0.92], w_pad=3.0)
    save(fig, "figure1_confidence_only_regenerated")


def figure2_all_models() -> None:
    data = pd.read_csv(ALL_MODEL_SUMMARY).sort_values("balanced_relative_score")
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), sharey=True)

    speed = data["mean_per_flight_speed_mae_mps"].to_numpy()
    speed_low = data["speed_ci_low"].to_numpy()
    speed_high = data["speed_ci_high"].to_numpy()
    speed_bars = axes[0].barh(y, speed, color="#2878B5")
    axes[0].errorbar(speed, y, xerr=[speed - speed_low, speed_high - speed],
                     fmt="none", ecolor="#222222", capsize=2, linewidth=1)
    axes[0].set_xlabel("Flight-balanced speed MAE (m/s)")
    axes[0].set_title("Wind-speed error - all common rows", weight="bold")

    direction = data["mean_per_flight_direction_mae_deg"].to_numpy()
    direction_low = data["direction_ci_low"].to_numpy()
    direction_high = data["direction_ci_high"].to_numpy()
    direction_bars = axes[1].barh(y, direction, color="#D95319")
    axes[1].errorbar(direction, y,
                     xerr=[direction - direction_low, direction_high - direction],
                     fmt="none", ecolor="#222222", capsize=2, linewidth=1)
    axes[1].set_xlabel("Flight-balanced circular MAE (degrees)")
    axes[1].set_title("Direction error - true speed > 1 m/s", weight="bold")

    labels = data["display_name"].replace({"Dummy": "Dummy baseline"}).tolist()
    for ax in axes:
        ax.set_yticks(y, labels=labels)
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelleft=True)
    axes[0].invert_yaxis()
    # Reserve room beyond the confidence intervals for numeric labels.
    axes[0].set_xlim(0, speed_high.max() * 1.16)
    axes[1].set_xlim(0, direction_high.max() * 1.16)
    for ax, bars, values, upper_ci, fmt in [
        (axes[0], speed_bars, speed, speed_high, "{:.3f}"),
        (axes[1], direction_bars, direction, direction_high, "{:.1f}"),
    ]:
        pad = ax.get_xlim()[1] * 0.012
        for bar, value, upper in zip(bars, values, upper_ci):
            ax.text(
                upper + pad,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=8.3,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.5},
            )
    fig.suptitle("Flight-Balanced Performance on Held-Out DJI Flights",
                 fontsize=15, weight="bold")
    fig.text(0.5, 0.015,
             "Shared row order: balanced speed-direction score (best to worst). Error bars: 95% bootstrap CIs; 52 eligible flights weighted equally.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0.02, 0.055, 0.99, 0.93], w_pad=3.2)
    save(fig, "figure2_all_models_labeled_regenerated")
    # A separately named final version makes the latest poster asset unambiguous.
    source_png = OUT / "figure2_all_models_labeled_regenerated.png"
    source_pdf = OUT / "figure2_all_models_labeled_regenerated.pdf"
    shutil.copy2(source_png, OUT / "figure2_all_models_values_final.png")
    shutil.copy2(source_pdf, OUT / "figure2_all_models_values_final.pdf")


def figure3_primary_residual() -> None:
    data = pd.read_csv(PRIMARY_PREDICTIONS)
    flight = data[data["_Group_ID"].astype(str) == "DJI_primary::272"].copy()
    if flight.empty:
        raise RuntimeError("Primary representative flight 272 was not found")
    flight = flight.sort_values("Sample_index")
    x = flight["Sample_index"].to_numpy()
    residual = circular_residual(flight["Predicted_wind_angle"], flight["Wind_angle"])

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.1), sharex=True)
    axes[0].plot(x, flight["Wind_speed"], label="Measured", color="#222222", linewidth=1.4)
    axes[0].plot(x, flight["Predicted_wind_speed"], label="Extra Trees", color="#2878B5", linewidth=1.2)
    axes[0].set_ylabel("Wind speed (m/s)")
    axes[0].legend(ncol=2, frameon=False)
    axes[0].set_title("Wind speed", weight="bold")

    axes[1].axhline(0, color="#222222", linewidth=1)
    axes[1].plot(x, residual, color="#D95319", linewidth=1.1)
    axes[1].set_ylim(-180, 180)
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    axes[1].set_ylabel("Signed circular residual (degrees)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_title("Direction residual: predicted minus measured", weight="bold")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle("Extra Trees on Held-Out DJI Flight 272", fontsize=15, weight="bold")
    fig.tight_layout(rect=[0.02, 0.03, 0.99, 0.94])
    save(fig, "figure3_primary_flight272_circular_residual_regenerated")

    segment_rows = []
    for segment_name, indices in zip(
        ["first third", "middle third", "final third"], np.array_split(np.arange(len(flight)), 3)
    ):
        segment_rows.append({
            "group_id": "DJI_primary::272", "segment": segment_name, "rows": len(indices),
            "speed_mae_mps": flight["Speed_error_mps"].abs().to_numpy()[indices].mean(),
            "direction_signed_bias_deg": residual[indices].mean(),
            "direction_circular_mae_deg": np.abs(residual[indices]).mean(),
        })
    segment_rows.insert(0, {
        "group_id": "DJI_primary::272", "segment": "entire flight", "rows": len(flight),
        "speed_mae_mps": flight["Speed_error_mps"].abs().mean(),
        "direction_signed_bias_deg": np.mean(residual),
        "direction_circular_mae_deg": np.mean(np.abs(residual)),
    })
    pd.DataFrame(segment_rows).to_csv(
        OUT / "figure3_primary_flight272_metrics.csv", index=False
    )


def select_median_amovfly_flight(data: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    data = data.copy()
    data["abs_speed_error"] = (
        data["Predicted_wind_speed"] - data["Wind_speed"]
    ).abs()
    table = data.groupby("_Group_ID").agg(
        rows=("_Group_ID", "size"), speed_mae_mps=("abs_speed_error", "mean")
    ).reset_index()
    eligible = table[table["rows"] >= 100].copy()
    median = eligible["speed_mae_mps"].median()
    eligible["distance_from_median"] = (eligible["speed_mae_mps"] - median).abs()
    selected = eligible.sort_values(["distance_from_median", "rows"], ascending=[True, False]).iloc[0]
    return str(selected["_Group_ID"]), table.sort_values("speed_mae_mps")


def figure4_amovfly_median() -> None:
    data = pd.read_csv(AMOVFLY_PREDICTIONS)
    group_id, table = select_median_amovfly_flight(data)
    flight = data[data["_Group_ID"].astype(str) == group_id].sort_values("_External_row_id")
    x = np.arange(len(flight))
    residual = circular_residual(flight["Predicted_wind_angle"], flight["Wind_angle"])
    flight_mae = np.mean(np.abs(flight["Predicted_wind_speed"] - flight["Wind_speed"]))
    dataset_mae = np.mean(np.abs(data["Predicted_wind_speed"] - data["Wind_speed"]))

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.1), sharex=True)
    axes[0].plot(x, flight["Wind_speed"], label="Measured", color="#222222", linewidth=1.35)
    axes[0].plot(x, flight["Predicted_wind_speed"], label="Extra Trees", color="#2878B5", linewidth=1.15)
    axes[0].set_ylabel("Wind speed (m/s)")
    axes[0].set_title(f"Wind speed - flight MAE {flight_mae:.2f} m/s", weight="bold")
    axes[0].legend(ncol=2, frameon=False)
    axes[1].axhline(0, color="#222222", linewidth=1)
    axes[1].plot(x, residual, color="#D95319", linewidth=1.0)
    axes[1].set_ylim(-180, 180)
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    axes[1].set_ylabel("Uncalibrated circular residual (degrees)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_title("Direction residual: predicted minus AMOVFLY label", weight="bold")
    for ax in axes:
        ax.grid(alpha=0.22)
    fig.suptitle(f"Extra Trees on Median-Error AMOVFLY Flight {group_id.split('::')[-1]}",
                 fontsize=14, weight="bold")
    fig.text(0.5, 0.012, f"Median flight selected objectively from flights with at least 100 rows; dataset-level speed MAE = {dataset_mae:.2f} m/s.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0.02, 0.045, 0.99, 0.93])
    save(fig, "figure4_amovfly_median_flight_regenerated")
    table.to_csv(OUT / "amovfly_per_flight_speed_mae.csv", index=False)
    pd.DataFrame({"selected_group_id": [group_id], "rows": [len(flight)],
                  "flight_speed_mae_mps": [flight_mae],
                  "dataset_speed_mae_mps": [dataset_mae],
                  "eligible_flight_median_speed_mae_mps": [table.loc[table["rows"] >= 100, "speed_mae_mps"].median()]}) \
        .to_csv(OUT / "figure4_amovfly_median_flight_metrics.csv", index=False)


def best_offset(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = circular_residual(predicted, observed)
    coarse = np.arange(-180.0, 180.0, 0.5)
    maes = np.array([np.mean(np.abs((residual + offset + 180) % 360 - 180)) for offset in coarse])
    return float(coarse[np.argmin(maes)])


def calibration_and_chance_analysis() -> None:
    amov = pd.read_csv(AMOVFLY_PREDICTIONS)
    amov = amov[amov["Wind_speed"] > 1.0].copy()
    groups = np.array(sorted(amov["_Group_ID"].astype(str).unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(groups)
    calibration_groups = set(groups[: len(groups) // 2])
    calibration = amov[amov["_Group_ID"].astype(str).isin(calibration_groups)]
    held_out = amov[~amov["_Group_ID"].astype(str).isin(calibration_groups)].copy()
    offset = best_offset(calibration["Wind_angle"].to_numpy(), calibration["Predicted_wind_angle"].to_numpy())
    raw = np.abs(circular_residual(held_out["Predicted_wind_angle"], held_out["Wind_angle"]))
    corrected_prediction = (held_out["Predicted_wind_angle"].to_numpy() + offset) % 360
    corrected = np.abs(circular_residual(corrected_prediction, held_out["Wind_angle"]))

    drone = pd.read_csv(DRONE_ONBOARD_PREDICTIONS)
    error = np.abs(circular_residual(drone["Predicted_wind_angle"], drone["Wind_angle"]))
    training_max = 14.9
    in_range = drone["Wind_speed"].to_numpy() <= training_max
    chance_within_15 = 30.0 / 360.0

    metrics = pd.DataFrame([
        {"analysis": "AMOVFLY held-out raw", "rows": len(raw), "direction_mae_deg": raw.mean(), "within_15_fraction": np.mean(raw <= 15), "fitted_offset_deg": 0.0},
        {"analysis": "AMOVFLY held-out offset-corrected", "rows": len(corrected), "direction_mae_deg": corrected.mean(), "within_15_fraction": np.mean(corrected <= 15), "fitted_offset_deg": offset},
        {"analysis": "Drone Onboard all rows", "rows": len(error), "direction_mae_deg": error.mean(), "within_15_fraction": np.mean(error <= 15), "fitted_offset_deg": np.nan},
        {"analysis": "Drone Onboard within DJI speed range", "rows": int(in_range.sum()), "direction_mae_deg": error[in_range].mean(), "within_15_fraction": np.mean(error[in_range] <= 15), "fitted_offset_deg": np.nan},
        {"analysis": "Uniform random direction chance", "rows": np.nan, "direction_mae_deg": 90.0, "within_15_fraction": chance_within_15, "fitted_offset_deg": np.nan},
    ])
    metrics.to_csv(OUT / "external_direction_calibration_and_chance.csv", index=False)
    (OUT / "amovfly_offset_split.txt").write_text(
        "Calibration was fit on a deterministic random half of whole AMOVFLY flights (seed 42) "
        "and evaluated on the disjoint remaining flights. Rows with true wind speed <= 1 m/s "
        "were excluded from direction analysis. The single offset minimizes calibration-set circular MAE.\n"
        f"Calibration flights: {len(calibration_groups)}; held-out flights: {len(groups) - len(calibration_groups)}; offset: {offset:.2f} degrees.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure1_confidence_only()
    figure2_all_models()
    figure3_primary_residual()
    figure4_amovfly_median()
    calibration_and_chance_analysis()
    print(f"Saved regenerated poster materials to {OUT}")


if __name__ == "__main__":
    main()
