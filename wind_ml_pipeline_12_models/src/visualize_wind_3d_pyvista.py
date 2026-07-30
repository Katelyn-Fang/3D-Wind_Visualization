#!/usr/bin/env python3
"""Interactive 3D wind visualization for the BU RISE wind-model pipeline.

This script reads a model run's ``test_predictions.csv`` and draws wind arrows
at the drone's 3D positions using PyVista. It can display measured wind,
predicted wind, or a linked side-by-side comparison.

Coordinate and direction assumptions
------------------------------------
* X is local east, Y is local north, and Z is altitude/up.
* Wind_angle and Predicted_wind_angle are meteorological directions in degrees:
  0° means wind coming FROM north and 90° means wind coming FROM east.
* The current model predicts horizontal wind speed/direction only, so arrows
  have zero vertical component unless ``--vertical-wind-column`` is supplied.

Examples
--------
Interactive comparison for one flight::

    python visualize_wind_3d_pyvista.py \
        --predictions results/scc/baseline/cpu/1_extra_trees/test_predictions.csv \
        --flight-id flight_01 --mode both

Save a high-resolution PNG instead of opening a window::

    python visualize_wind_3d_pyvista.py \
        --predictions test_predictions.csv \
        --mode predicted --output predicted_wind_3d.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TRUE_COLUMNS = ("Wind_speed", "Wind_angle")
PREDICTED_COLUMNS = ("Predicted_wind_speed", "Predicted_wind_angle")
POSITION_COLUMNS = ("X", "Y", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to a model run's test_predictions.csv.",
    )
    parser.add_argument(
        "--mode",
        choices=("true", "predicted", "both"),
        default="both",
        help="Wind field to display. 'both' creates linked side-by-side views.",
    )
    parser.add_argument(
        "--flight-id",
        default=None,
        help="Optional exact Flight_ID to display.",
    )
    parser.add_argument(
        "--source-dataset",
        default=None,
        help="Optional exact Source_dataset value to display.",
    )
    parser.add_argument(
        "--group-id",
        default=None,
        help="Optional exact _Group_ID to display.",
    )
    parser.add_argument(
        "--list-flights",
        action="store_true",
        help="Print available flight/group identifiers and exit.",
    )
    parser.add_argument(
        "--max-arrows",
        type=int,
        default=500,
        help="Maximum arrows across the selected data. Downsampling is even within each flight.",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=0.0,
        help="Hide arrows whose displayed wind speed is below this value in m/s.",
    )
    parser.add_argument(
        "--arrow-scale",
        type=float,
        default=None,
        help="Arrow-length multiplier in coordinate units per m/s. Default is chosen automatically.",
    )
    parser.add_argument(
        "--vertical-exaggeration",
        type=float,
        default=1.0,
        help="Multiply Z coordinates by this value for display only.",
    )
    parser.add_argument(
        "--vertical-wind-column",
        default=None,
        help="Optional column containing upward wind velocity in m/s.",
    )
    parser.add_argument(
        "--angles-point-to",
        action="store_true",
        help="Treat angle columns as the direction wind points TO rather than meteorological FROM.",
    )
    parser.add_argument(
        "--x-is-north",
        action="store_true",
        help="Use X=north and Y=east instead of the default X=east and Y=north.",
    )
    parser.add_argument(
        "--color-map",
        default="viridis",
        help="Matplotlib/VTK colormap used for wind speed.",
    )
    parser.add_argument(
        "--theme",
        choices=("dark", "light"),
        default="dark",
        help="Background and annotation theme.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
        help="Size of drone-position points.",
    )
    parser.add_argument(
        "--path-width",
        type=float,
        default=3.0,
        help="Width of each drone flight-path line.",
    )
    parser.add_argument(
        "--camera",
        choices=("isometric", "top", "xy", "xz", "yz"),
        default="isometric",
        help="Initial camera orientation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional PNG/JPG screenshot path. Supplying it enables off-screen rendering.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Optional interactive HTML output. Requires the optional 'trame' package.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Rendering-window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Rendering-window height in pixels.",
    )
    parser.add_argument(
        "--screenshot-scale",
        type=int,
        default=2,
        help="Resolution multiplier used when saving a screenshot.",
    )
    return parser


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + ". Use a test_predictions.csv produced by the v0.3.0 pipeline."
        )


def list_flights(frame: pd.DataFrame) -> None:
    columns = [
        column
        for column in ("_Group_ID", "Source_dataset", "Flight_ID")
        if column in frame.columns
    ]
    if not columns:
        print("No flight identifier columns were found.")
        return
    summary = frame[columns].drop_duplicates().sort_values(columns, kind="stable")
    print(summary.to_string(index=False))


def filter_rows(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = frame.copy()
    filters = (
        ("Flight_ID", args.flight_id),
        ("Source_dataset", args.source_dataset),
        ("_Group_ID", args.group_id),
    )
    for column, value in filters:
        if value is None:
            continue
        if column not in selected.columns:
            raise ValueError(f"Cannot filter by {column}: the column is not present.")
        selected = selected[selected[column].astype(str) == str(value)]

    if selected.empty:
        raise ValueError("The selected flight filters produced zero rows.")
    return selected.reset_index(drop=True)


def group_column(frame: pd.DataFrame) -> str | None:
    for column in ("_Group_ID", "Flight_ID", "Source_dataset"):
        if column in frame.columns:
            return column
    return None


def sort_group(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("Elapsed_s", "Sample_index", "Timestamp"):
        if column in frame.columns:
            if column == "Timestamp":
                temporary = pd.to_datetime(frame[column], errors="coerce")
                if temporary.notna().any():
                    return frame.assign(_sort_time=temporary).sort_values(
                        "_sort_time", kind="stable"
                    ).drop(columns="_sort_time")
            else:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                if numeric.notna().any():
                    return frame.assign(_sort_value=numeric).sort_values(
                        "_sort_value", kind="stable"
                    ).drop(columns="_sort_value")
    return frame


def evenly_downsample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum <= 0:
        raise ValueError("--max-arrows must be positive.")
    if len(frame) <= maximum:
        return frame.reset_index(drop=True)

    group = group_column(frame)
    if group is None:
        indices = np.linspace(0, len(frame) - 1, maximum, dtype=int)
        return frame.iloc[np.unique(indices)].reset_index(drop=True)

    pieces: list[pd.DataFrame] = []
    grouped = list(frame.groupby(group, sort=False, observed=True))
    counts = np.array([len(part) for _, part in grouped], dtype=float)
    quotas = np.maximum(1, np.floor(maximum * counts / counts.sum()).astype(int))

    # Adjust quotas so their sum is close to the requested maximum.
    while quotas.sum() > maximum and np.any(quotas > 1):
        index = int(np.argmax(quotas))
        quotas[index] -= 1
    while quotas.sum() < maximum:
        remaining = counts - quotas
        index = int(np.argmax(remaining))
        if remaining[index] <= 0:
            break
        quotas[index] += 1

    for (_, part), quota in zip(grouped, quotas, strict=True):
        ordered = sort_group(part)
        if len(ordered) <= quota:
            pieces.append(ordered)
        else:
            indices = np.linspace(0, len(ordered) - 1, quota, dtype=int)
            pieces.append(ordered.iloc[np.unique(indices)])
    return pd.concat(pieces, ignore_index=True)


def angles_to_vectors(
    speeds: np.ndarray,
    angles_deg: np.ndarray,
    *,
    vertical: np.ndarray | None,
    angles_point_to: bool,
    x_is_north: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert compass angles and speed to unit orientation vectors and magnitudes."""
    speed = np.asarray(speeds, dtype=float)
    angle = np.deg2rad(np.asarray(angles_deg, dtype=float) % 360.0)

    # Compass bearings are clockwise from north. These components point TO the
    # bearing. Meteorological angles describe where the wind comes FROM, so the
    # horizontal components are negated by default.
    east = speed * np.sin(angle)
    north = speed * np.cos(angle)
    if not angles_point_to:
        east = -east
        north = -north

    up = np.zeros_like(speed) if vertical is None else np.asarray(vertical, dtype=float)
    if x_is_north:
        vectors = np.column_stack((north, east, up))
    else:
        vectors = np.column_stack((east, north, up))

    magnitude = np.linalg.norm(vectors, axis=1)
    safe = np.where(magnitude > 1e-12, magnitude, 1.0)
    unit_vectors = vectors / safe[:, None]
    unit_vectors[magnitude <= 1e-12] = np.array([1.0, 0.0, 0.0])
    return unit_vectors, magnitude


def auto_arrow_scale(points: np.ndarray, speeds: np.ndarray) -> float:
    extent = np.ptp(points, axis=0)
    diagonal = float(np.linalg.norm(extent))
    positive = np.asarray(speeds, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    representative_speed = float(np.percentile(positive, 95)) if positive.size else 1.0
    if diagonal <= 0:
        diagonal = 10.0
    return 0.08 * diagonal / max(representative_speed, 1e-6)


def add_flight_paths(
    plotter,
    frame: pd.DataFrame,
    points: np.ndarray,
    path_width: float,
    *,
    path_color: str,
) -> None:
    import pyvista as pv

    group = group_column(frame)
    if group is None:
        groups = [("Flight path", frame.index.to_numpy())]
    else:
        groups = [
            (str(name), part.index.to_numpy())
            for name, part in frame.groupby(group, sort=False, observed=True)
        ]

    added_label = False
    for _, indices in groups:
        if len(indices) < 2:
            continue
        path = pv.lines_from_points(points[indices], close=False)
        plotter.add_mesh(
            path,
            color=path_color,
            line_width=path_width,
            opacity=0.65,
            render_lines_as_tubes=True,
            label="Drone path" if not added_label else None,
        )
        added_label = True


def add_wind_scene(
    plotter,
    frame: pd.DataFrame,
    *,
    speed_column: str,
    angle_column: str,
    title: str,
    args: argparse.Namespace,
    arrow_scale: float,
    color_limits: tuple[float, float],
    show_scalar_bar: bool,
) -> None:
    import pyvista as pv

    points = frame.loc[:, POSITION_COLUMNS].to_numpy(dtype=float)
    points[:, 2] *= args.vertical_exaggeration

    speeds = pd.to_numeric(frame[speed_column], errors="coerce").to_numpy(dtype=float)
    angles = pd.to_numeric(frame[angle_column], errors="coerce").to_numpy(dtype=float)
    vertical = None
    if args.vertical_wind_column:
        vertical = pd.to_numeric(
            frame[args.vertical_wind_column], errors="coerce"
        ).to_numpy(dtype=float)

    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(speeds)
        & np.isfinite(angles)
        & (speeds >= args.min_speed)
    )
    if vertical is not None:
        valid &= np.isfinite(vertical)

    points = points[valid]
    speeds = speeds[valid]
    angles = angles[valid]
    selected_frame = frame.loc[valid].reset_index(drop=True)
    selected_vertical = None if vertical is None else vertical[valid]

    if len(points) == 0:
        raise ValueError(
            f"No valid arrows remain for {title}. Lower --min-speed or check the input columns."
        )

    directions, magnitudes = angles_to_vectors(
        speeds,
        angles,
        vertical=selected_vertical,
        angles_point_to=args.angles_point_to,
        x_is_north=args.x_is_north,
    )

    cloud = pv.PolyData(points)
    cloud.point_data["wind_direction_vector"] = directions
    cloud.point_data["wind_speed_mps"] = magnitudes

    glyphs = cloud.glyph(
        orient="wind_direction_vector",
        scale="wind_speed_mps",
        factor=arrow_scale,
    )

    foreground = "white" if args.theme == "dark" else "black"
    path_color = "white" if args.theme == "dark" else "black"
    point_color = "white" if args.theme == "dark" else "dimgray"

    plotter.add_mesh(
        glyphs,
        scalars="wind_speed_mps",
        cmap=args.color_map,
        clim=color_limits,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={
            "title": "Speed (m/s)",
            "vertical": True,
            "position_x": 0.86,
            "position_y": 0.16,
            "height": 0.62,
            "width": 0.08,
            "title_font_size": 13,
            "label_font_size": 11,
            "color": foreground,
        },
        smooth_shading=True,
    )
    plotter.add_points(
        points,
        color=point_color,
        point_size=args.point_size,
        opacity=0.45,
        render_points_as_spheres=True,
    )
    add_flight_paths(
        plotter, selected_frame, points, args.path_width, path_color=path_color
    )

    x_title = "North / X (m)" if args.x_is_north else "East / X (m)"
    y_title = "East / Y (m)" if args.x_is_north else "North / Y (m)"
    z_title = "Altitude / Z (m)"
    if not np.isclose(args.vertical_exaggeration, 1.0):
        z_title += f" × {args.vertical_exaggeration:g}"

    plotter.add_text(
        title, position="upper_left", font_size=15, color=foreground
    )
    plotter.show_grid(
        xtitle=x_title,
        ytitle=y_title,
        ztitle=z_title,
        grid="back",
        location="outer",
        color=foreground,
    )
    plotter.add_axes()


def set_camera(plotter, camera: str) -> None:
    if camera == "isometric":
        plotter.view_isometric()
    elif camera in {"top", "xy"}:
        plotter.view_xy()
    elif camera == "xz":
        plotter.view_xz()
    elif camera == "yz":
        plotter.view_yz()
    plotter.reset_camera()


def speed_limits(frame: pd.DataFrame, mode: str, min_speed: float) -> tuple[float, float]:
    columns: list[str] = []
    if mode in {"true", "both"}:
        columns.append(TRUE_COLUMNS[0])
    if mode in {"predicted", "both"}:
        columns.append(PREDICTED_COLUMNS[0])

    arrays = [pd.to_numeric(frame[column], errors="coerce").to_numpy() for column in columns]
    values = np.concatenate(arrays)
    values = values[np.isfinite(values) & (values >= min_speed)]
    if values.size == 0:
        return (0.0, 1.0)
    low = max(0.0, float(np.min(values)))
    high = float(np.percentile(values, 99))
    if high <= low:
        high = low + 1.0
    return low, high


def main() -> int:
    args = build_parser().parse_args()
    if args.vertical_exaggeration <= 0:
        raise ValueError("--vertical-exaggeration must be positive.")
    if args.screenshot_scale < 1:
        raise ValueError("--screenshot-scale must be at least 1.")
    if not args.predictions.exists():
        raise FileNotFoundError(f"Could not find {args.predictions.resolve()}")

    frame = pd.read_csv(args.predictions)
    require_columns(frame, POSITION_COLUMNS)

    if args.list_flights:
        list_flights(frame)
        return 0

    if args.mode in {"true", "both"}:
        require_columns(frame, TRUE_COLUMNS)
    if args.mode in {"predicted", "both"}:
        require_columns(frame, PREDICTED_COLUMNS)
    if args.vertical_wind_column:
        require_columns(frame, (args.vertical_wind_column,))

    frame = filter_rows(frame, args)
    frame = evenly_downsample(frame, args.max_arrows)

    points = frame.loc[:, POSITION_COLUMNS].to_numpy(dtype=float)
    points[:, 2] *= args.vertical_exaggeration
    scale_speeds = []
    if args.mode in {"true", "both"}:
        scale_speeds.append(pd.to_numeric(frame[TRUE_COLUMNS[0]], errors="coerce").to_numpy())
    if args.mode in {"predicted", "both"}:
        scale_speeds.append(
            pd.to_numeric(frame[PREDICTED_COLUMNS[0]], errors="coerce").to_numpy()
        )
    all_speeds = np.concatenate(scale_speeds)
    arrow_scale = (
        args.arrow_scale
        if args.arrow_scale is not None
        else auto_arrow_scale(points, all_speeds)
    )
    color_limits = speed_limits(frame, args.mode, args.min_speed)

    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "PyVista is not installed. Run: pip install pyvista"
        ) from error

    off_screen = args.output is not None and args.html_output is None
    shape = (1, 2) if args.mode == "both" else (1, 1)
    plotter = pv.Plotter(
        shape=shape,
        off_screen=off_screen,
        window_size=(args.window_width, args.window_height),
    )
    background = "#15171a" if args.theme == "dark" else "white"
    plotter.set_background(background)

    if args.mode in {"true", "both"}:
        plotter.subplot(0, 0)
        add_wind_scene(
            plotter,
            frame,
            speed_column=TRUE_COLUMNS[0],
            angle_column=TRUE_COLUMNS[1],
            title="Measured wind",
            args=args,
            arrow_scale=arrow_scale,
            color_limits=color_limits,
            show_scalar_bar=args.mode != "both",
        )
        set_camera(plotter, args.camera)

    if args.mode in {"predicted", "both"}:
        column = 1 if args.mode == "both" else 0
        plotter.subplot(0, column)
        add_wind_scene(
            plotter,
            frame,
            speed_column=PREDICTED_COLUMNS[0],
            angle_column=PREDICTED_COLUMNS[1],
            title="Predicted wind",
            args=args,
            arrow_scale=arrow_scale,
            color_limits=color_limits,
            show_scalar_bar=True,
        )
        set_camera(plotter, args.camera)

    if args.mode == "both":
        plotter.link_views()

    flight_description = args.flight_id or args.group_id or "selected test flights"
    print(f"Rows plotted: {len(frame):,}")
    print(f"Selection: {flight_description}")
    print(f"Arrow scale: {arrow_scale:.6g} coordinate units per m/s")
    print(f"Color limits: {color_limits[0]:.3g}–{color_limits[1]:.3g} m/s")

    if args.html_output is not None:
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        # Render once before export so the actors and cameras are initialized.
        plotter.render()
        try:
            plotter.export_html(args.html_output)
        except ImportError as error:
            raise RuntimeError(
                "HTML export requires trame. Run: pip install 'pyvista[jupyter]' trame"
            ) from error
        print(f"Saved interactive HTML: {args.html_output.resolve()}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plotter.show(
            screenshot=str(args.output),
            auto_close=False,
            interactive=False,
        )
        if args.screenshot_scale > 1:
            plotter.screenshot(
                str(args.output),
                scale=args.screenshot_scale,
                return_img=False,
            )
        plotter.close()
        print(f"Saved screenshot: {args.output.resolve()}")
    elif args.html_output is None:
        plotter.show(title="BU RISE 3D Wind Visualization")
    else:
        plotter.close()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)
