"""
UNIVERSAL DRONE DATA STANDARDIZER

This script creates these standardized columns:

    Timestamp
    X
    Y
    Z
    Roll
    Pitch
    Yaw
    Wind_speed
    Wind_angle
    Source_dataset
    Battery_V
    Battery_C

It also preserves:

    Flight_ID
    Elapsed_s
    Coordinate_frame

Conventions:
    Roll, Pitch, and Yaw are stored in radians.
    Wind_angle is stored in degrees from 0 to less than 360.
    Battery_V is battery voltage in volts.
    Battery_C is battery current in amperes.

The script:
    - Recognizes many common drone-data column names
    - Converts text measurements to numbers
    - Creates timestamps
    - Converts quaternions to roll, pitch, and yaw
    - Derives wind speed and angle from wind-vector components
    - Merges an optional flight-parameters file
    - Removes exact duplicate rows
    - Reports missing required columns

The script does NOT:
    - Invent missing measurements
    - Interpolate missing values
    - Smooth the data
    - Remove statistical outliers
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

# Main telemetry file
TELEMETRY_FILE = "flights_primary.csv"

# Optional flight-parameter or metadata file.
# Set this to None when there is no separate metadata file.
METADATA_FILE = "parameters_primary.csv"

# Name of the cleaned output file
OUTPUT_FILE = "DJI_primary_standardized.csv"

# Label placed in Source_dataset for every row
SOURCE_DATASET = "DJI_primary"

# Use:
#   "auto"     when you do not know the angle units
#   "degrees"  when roll, pitch, and yaw are definitely degrees
#   "radians"  when roll, pitch, and yaw are definitely radians
ATTITUDE_UNIT = "auto"

# Use:
#   "auto"     when the column header states the unit
#   "degrees"  when wind angle is definitely degrees
#   "radians"  when wind angle is definitely radians
WIND_ANGLE_UNIT = "auto"

# True keeps every original column after the standardized columns.
# False creates a smaller standardized output.
KEEP_ORIGINAL_COLUMNS = False


# ============================================================
# STANDARDIZED OUTPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "Timestamp",
    "X",
    "Y",
    "Z",
    "Roll",
    "Pitch",
    "Yaw",
    "Wind_speed",
    "Wind_angle",
    "Source_dataset",
    "Battery_V",
    "Battery_C",
]

OPTIONAL_COLUMNS = [
    "Flight_ID",
    "Elapsed_s",
    "Coordinate_frame",
]


# ============================================================
# COMMON COLUMN-NAME ALIASES
#
# Add a dataset's unusual headers here when necessary.
# All names are normalized before matching.
# ============================================================

ALIASES = {
    "flight_id": [
        "flight",
        "flight_id",
        "flight_number",
        "flight_no",
        "run",
        "run_id",
        "trial",
        "trial_id",
        "mission",
        "mission_id",
        "experiment",
        "experiment_id",
    ],

    "timestamp": [
        "timestamp",
        "time_stamp",
        "datetime",
        "date_time",
        "utc_timestamp",
        "timestamp_utc",
        "recorded_at",
        "created_at",
        "ros_time",
        "unix_time",
        "epoch_time",
    ],

    "date": [
        "date",
        "flight_date",
        "recording_date",
        "experiment_date",
    ],

    "clock_time": [
        "time_day",
        "clock_time",
        "time_of_day",
        "local_time",
        "start_time",
        "flight_start_time",
    ],

    "elapsed_time": [
        "elapsed",
        "elapsed_s",
        "elapsed_sec",
        "elapsed_seconds",
        "elapsed_time",
        "elapsed_time_s",
        "flight_time",
        "flight_time_s",
        "relative_time",
        "relative_time_s",
        "seconds",
        "time_s",
    ],

    "x": [
        "x",
        "x_m",
        "position_x",
        "position_x_m",
        "pos_x",
        "local_x",
        "north",
        "north_m",
        "ned_x",
        "enu_x",
        "longitude",
        "longitude_deg",
        "lon",
        "lng",
        "gps_lon",
        "gps_longitude",
    ],

    "y": [
        "y",
        "y_m",
        "position_y",
        "position_y_m",
        "pos_y",
        "local_y",
        "east",
        "east_m",
        "ned_y",
        "enu_y",
        "latitude",
        "latitude_deg",
        "lat",
        "gps_lat",
        "gps_latitude",
    ],

    "z": [
        "z",
        "z_m",
        "position_z",
        "position_z_m",
        "pos_z",
        "local_z",
        "altitude",
        "altitude_m",
        "height",
        "height_m",
        "gps_altitude",
        "down",
        "down_m",
        "ned_z",
        "enu_z",
    ],

    "roll": [
        "roll",
        "roll_rad",
        "roll_deg",
        "roll_angle",
        "roll_angle_rad",
        "roll_angle_deg",
        "attitude_roll",
        "phi",
    ],

    "pitch": [
        "pitch",
        "pitch_rad",
        "pitch_deg",
        "pitch_angle",
        "pitch_angle_rad",
        "pitch_angle_deg",
        "attitude_pitch",
        "theta",
    ],

    "yaw": [
        "yaw",
        "yaw_rad",
        "yaw_deg",
        "yaw_angle",
        "yaw_angle_rad",
        "yaw_angle_deg",
        "attitude_yaw",
        "psi",
        "heading",
        "heading_rad",
        "heading_deg",
    ],

    "qx": [
        "qx",
        "q_x",
        "quat_x",
        "quaternion_x",
        "orientation_x",
    ],

    "qy": [
        "qy",
        "q_y",
        "quat_y",
        "quaternion_y",
        "orientation_y",
    ],

    "qz": [
        "qz",
        "q_z",
        "quat_z",
        "quaternion_z",
        "orientation_z",
    ],

    "qw": [
        "qw",
        "q_w",
        "quat_w",
        "quaternion_w",
        "orientation_w",
    ],

    "wind_speed": [
        "wind_speed",
        "wind_speed_mps",
        "wind_speed_m_s",
        "windspeed",
        "wind_velocity",
        "wind_magnitude",
        "wind_magnitude_mps",
        "wind_magnitude_m_s",
    ],

    "wind_angle": [
        "wind_angle",
        "wind_angle_deg",
        "wind_angle_rad",
        "wind_direction",
        "wind_direction_deg",
        "wind_direction_rad",
        "wind_dir",
        "wind_bearing",
        "wind_heading",
    ],

    # Wind-component convention used below:
    # u = eastward velocity
    # v = northward velocity
    # w = upward velocity
    "wind_u": [
        "wind_u",
        "wind_u_mps",
        "u_wind",
        "u_component",
        "wind_east",
        "wind_east_mps",
        "wind_x",
        "wind_x_mps",
    ],

    "wind_v": [
        "wind_v",
        "wind_v_mps",
        "v_wind",
        "v_component",
        "wind_north",
        "wind_north_mps",
        "wind_y",
        "wind_y_mps",
    ],

    "wind_w": [
        "wind_w",
        "wind_w_mps",
        "w_wind",
        "w_component",
        "wind_vertical",
        "vertical_wind",
        "wind_z",
        "wind_z_mps",
    ],

    "battery_voltage": [
        "battery_v",
        "battery_voltage",
        "battery_voltage_v",
        "voltage",
        "voltage_v",
        "bat_voltage",
        "vbat",
        "battery_volts",
    ],

    "battery_current": [
        "battery_c",
        "battery_current",
        "battery_current_a",
        "current",
        "current_a",
        "bat_current",
        "ibat",
        "battery_amps",
    ],
}


# ============================================================
# COLUMN-NAME FUNCTIONS
# ============================================================

def normalize_column_name(name):
    """
    Convert a column name into a consistent lowercase format.

    Example:
        Wind Speed (m/s) -> wind_speed_m_s
    """

    name = str(name).strip().lower()
    name = name.replace("°", "_deg")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)

    return name.strip("_")


def normalize_all_headers(df):
    """
    Normalize every header and ensure duplicate headers remain unique.
    """

    df = df.copy()

    normalized_headers = []
    header_counts = {}

    for original_header in df.columns:
        base_header = normalize_column_name(original_header)

        count = header_counts.get(base_header, 0)
        header_counts[base_header] = count + 1

        if count == 0:
            normalized_headers.append(base_header)
        else:
            normalized_headers.append(
                f"{base_header}_{count + 1}"
            )

    df.columns = normalized_headers

    return df


def find_column(df, alias_group):
    """
    Return the first column matching one of the aliases.
    """

    for possible_name in ALIASES[alias_group]:
        normalized_name = normalize_column_name(possible_name)

        if normalized_name in df.columns:
            return normalized_name

    return None


def numeric_series(df, column_name):
    """
    Convert one column to numeric values.

    Invalid values become NaN.
    """

    if column_name is None:
        return pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

    return pd.to_numeric(
        df[column_name],
        errors="coerce",
    ).astype("float64")


# ============================================================
# OPTIONAL METADATA MERGING
# ============================================================

def merge_metadata(telemetry_df, metadata_df):
    """
    Merge flight-level metadata into telemetry using a flight ID.

    The merge occurs only when both files contain a recognizable
    flight identifier.
    """

    telemetry_flight_column = find_column(
        telemetry_df,
        "flight_id",
    )

    metadata_flight_column = find_column(
        metadata_df,
        "flight_id",
    )

    if telemetry_flight_column is None:
        print(
            "Metadata not merged: no flight ID was found "
            "in the telemetry file."
        )
        return telemetry_df

    if metadata_flight_column is None:
        print(
            "Metadata not merged: no flight ID was found "
            "in the metadata file."
        )
        return telemetry_df

    if metadata_flight_column != telemetry_flight_column:
        metadata_df = metadata_df.rename(
            columns={
                metadata_flight_column:
                telemetry_flight_column
            }
        )

    # There should generally be one metadata row per flight.
    metadata_df = metadata_df.drop_duplicates(
        subset=[telemetry_flight_column],
        keep="first",
    )

    merged_df = telemetry_df.merge(
        metadata_df,
        on=telemetry_flight_column,
        how="left",
        suffixes=("", "_metadata"),
    )

    # Fill gaps in telemetry columns using matching metadata columns.
    for column in list(merged_df.columns):
        if not column.endswith("_metadata"):
            continue

        original_name = column.removesuffix("_metadata")

        if original_name in merged_df.columns:
            merged_df[original_name] = (
                merged_df[original_name]
                .combine_first(merged_df[column])
            )

            merged_df = merged_df.drop(
                columns=[column]
            )

        else:
            merged_df = merged_df.rename(
                columns={column: original_name}
            )

    print("Metadata successfully merged by flight ID.")

    return merged_df


# ============================================================
# TIMESTAMP FUNCTIONS
# ============================================================

def parse_epoch_timestamp(series):
    """
    Parse timestamps expressed as Unix seconds, milliseconds,
    microseconds, or nanoseconds.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid_values = values.dropna()

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    if valid_values.empty:
        return result

    median_value = float(
        valid_values.abs().median()
    )

    if median_value >= 1e17:
        return pd.to_datetime(
            values,
            unit="ns",
            errors="coerce",
        )

    if median_value >= 1e14:
        return pd.to_datetime(
            values,
            unit="us",
            errors="coerce",
        )

    if median_value >= 1e11:
        return pd.to_datetime(
            values,
            unit="ms",
            errors="coerce",
        )

    if median_value >= 1e9:
        return pd.to_datetime(
            values,
            unit="s",
            errors="coerce",
        )

    # Smaller values are likely elapsed seconds, not dates.
    return result


def construct_timestamp(df):
    """
    Create the Timestamp and Elapsed_s columns.

    Priority:
        1. Existing timestamp column
        2. Date + clock time + elapsed seconds
        3. Date + elapsed seconds
    """

    mapping = {}

    elapsed_column = find_column(
        df,
        "elapsed_time",
    )

    elapsed_seconds = numeric_series(
        df,
        elapsed_column,
    )

    if elapsed_column is not None:
        mapping["Elapsed_s"] = elapsed_column

    timestamp_column = find_column(
        df,
        "timestamp",
    )

    timestamp = pd.Series(
        pd.NaT,
        index=df.index,
        dtype="datetime64[ns]",
    )

    # Try a direct timestamp first.
    if timestamp_column is not None:
        source_values = df[timestamp_column]

        if pd.api.types.is_numeric_dtype(source_values):
            timestamp = parse_epoch_timestamp(
                source_values
            )
        else:
            timestamp = pd.to_datetime(
                source_values,
                errors="coerce",
            )

        mapping["Timestamp"] = timestamp_column

    date_column = find_column(
        df,
        "date",
    )

    clock_column = find_column(
        df,
        "clock_time",
    )

    # Construct date + clock + elapsed.
    if date_column is not None and clock_column is not None:
        base_time_text = (
            df[date_column]
            .astype("string")
            .str.strip()
            + " "
            + df[clock_column]
            .astype("string")
            .str.strip()
        )

        base_timestamp = pd.to_datetime(
            base_time_text,
            errors="coerce",
        )

        constructed_timestamp = (
            base_timestamp
            + pd.to_timedelta(
                elapsed_seconds.fillna(0),
                unit="s",
            )
        )

        timestamp = timestamp.combine_first(
            constructed_timestamp
        )

        mapping["Timestamp"] = (
            f"{date_column} + "
            f"{clock_column} + "
            f"{elapsed_column or '0 seconds'}"
        )

    # Construct date + elapsed if clock time is unavailable.
    elif date_column is not None:
        base_timestamp = pd.to_datetime(
            df[date_column],
            errors="coerce",
        )

        constructed_timestamp = (
            base_timestamp
            + pd.to_timedelta(
                elapsed_seconds.fillna(0),
                unit="s",
            )
        )

        timestamp = timestamp.combine_first(
            constructed_timestamp
        )

        mapping["Timestamp"] = (
            f"{date_column} + "
            f"{elapsed_column or '0 seconds'}"
        )

    return timestamp, elapsed_seconds, mapping


# ============================================================
# ATTITUDE FUNCTIONS
# ============================================================

def convert_attitude_to_radians(
    values,
    source_column,
    attitude_unit,
):
    """
    Convert Euler angles to radians.
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).astype("float64")

    if attitude_unit not in {
        "auto",
        "degrees",
        "radians",
    }:
        raise ValueError(
            "ATTITUDE_UNIT must be auto, degrees, or radians."
        )

    should_convert = False

    if attitude_unit == "degrees":
        should_convert = True

    elif attitude_unit == "radians":
        should_convert = False

    else:
        source_name = source_column or ""

        if "deg" in source_name:
            should_convert = True

        elif "rad" in source_name:
            should_convert = False

        else:
            # Conservative magnitude check.
            # Typical radian angles are normally below about 6.3.
            upper_value = values.abs().quantile(0.995)

            if (
                pd.notna(upper_value)
                and 8 < upper_value <= 720
            ):
                should_convert = True

    if should_convert:
        values = np.deg2rad(values)

    return values


def quaternion_to_euler(df):
    """
    Calculate roll, pitch, and yaw from quaternion x/y/z/w.

    Returns angles in radians.
    """

    qx_column = find_column(df, "qx")
    qy_column = find_column(df, "qy")
    qz_column = find_column(df, "qz")
    qw_column = find_column(df, "qw")

    quaternion_columns = [
        qx_column,
        qy_column,
        qz_column,
        qw_column,
    ]

    if any(column is None for column in quaternion_columns):
        empty = pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

        return (
            empty.copy(),
            empty.copy(),
            empty.copy(),
            {},
        )

    qx = numeric_series(df, qx_column)
    qy = numeric_series(df, qy_column)
    qz = numeric_series(df, qz_column)
    qw = numeric_series(df, qw_column)

    # Normalize each quaternion.
    magnitude = np.sqrt(
        qx ** 2
        + qy ** 2
        + qz ** 2
        + qw ** 2
    )

    valid_quaternion = magnitude > 1e-12

    qx = qx.where(valid_quaternion) / magnitude.where(
        valid_quaternion
    )

    qy = qy.where(valid_quaternion) / magnitude.where(
        valid_quaternion
    )

    qz = qz.where(valid_quaternion) / magnitude.where(
        valid_quaternion
    )

    qw = qw.where(valid_quaternion) / magnitude.where(
        valid_quaternion
    )

    # Quaternion to Euler conversion.
    roll = np.arctan2(
        2 * (qw * qx + qy * qz),
        1 - 2 * (qx ** 2 + qy ** 2),
    )

    pitch_input = (
        2 * (qw * qy - qz * qx)
    ).clip(-1, 1)

    pitch = np.arcsin(
        pitch_input
    )

    yaw = np.arctan2(
        2 * (qw * qz + qx * qy),
        1 - 2 * (qy ** 2 + qz ** 2),
    )

    source_description = (
        f"derived from {qx_column}, {qy_column}, "
        f"{qz_column}, {qw_column}"
    )

    mapping = {
        "Roll": source_description,
        "Pitch": source_description,
        "Yaw": source_description,
    }

    return roll, pitch, yaw, mapping


def get_roll_pitch_yaw(df):
    """
    Read Euler angles directly when available.

    If direct Euler angles are missing, calculate them from
    quaternion orientation.
    """

    mapping = {}

    quaternion_roll, quaternion_pitch, quaternion_yaw, quaternion_mapping = (
        quaternion_to_euler(df)
    )

    output_values = []

    attitude_information = [
        ("roll", "Roll", quaternion_roll),
        ("pitch", "Pitch", quaternion_pitch),
        ("yaw", "Yaw", quaternion_yaw),
    ]

    for alias_group, output_name, quaternion_values in attitude_information:
        direct_column = find_column(
            df,
            alias_group,
        )

        if direct_column is not None:
            direct_values = convert_attitude_to_radians(
                df[direct_column],
                direct_column,
                ATTITUDE_UNIT,
            )

            # Fill direct Euler gaps using quaternion-derived values.
            final_values = direct_values.combine_first(
                quaternion_values
            )

            mapping[output_name] = direct_column

            if (
                quaternion_values.notna().any()
                and direct_values.isna().any()
            ):
                mapping[output_name] += (
                    "; quaternion used for missing values"
                )

        else:
            final_values = quaternion_values

            if output_name in quaternion_mapping:
                mapping[output_name] = quaternion_mapping[
                    output_name
                ]

        output_values.append(final_values)

    return (
        output_values[0],
        output_values[1],
        output_values[2],
        mapping,
    )


# ============================================================
# WIND FUNCTIONS
# ============================================================

def convert_wind_angle_to_degrees(
    values,
    source_column,
):
    """
    Convert wind angle to degrees and normalize it to [0, 360).
    """

    values = pd.to_numeric(
        values,
        errors="coerce",
    ).astype("float64")

    if WIND_ANGLE_UNIT not in {
        "auto",
        "degrees",
        "radians",
    }:
        raise ValueError(
            "WIND_ANGLE_UNIT must be auto, degrees, or radians."
        )

    should_convert = False

    if WIND_ANGLE_UNIT == "radians":
        should_convert = True

    elif WIND_ANGLE_UNIT == "degrees":
        should_convert = False

    else:
        source_name = source_column or ""

        # For wind direction, avoid guessing based only on magnitude.
        # A valid degree dataset can contain only small angles.
        if "rad" in source_name:
            should_convert = True

    if should_convert:
        values = np.rad2deg(values)

    return values % 360


def get_wind_data(df):
    """
    Read wind speed and angle directly.

    When necessary, derive them from vector components:

        wind_u = eastward component
        wind_v = northward component
        wind_w = upward component

    Wind angle is meteorological direction: where the wind
    comes FROM.
    """

    mapping = {}

    wind_speed_column = find_column(
        df,
        "wind_speed",
    )

    wind_angle_column = find_column(
        df,
        "wind_angle",
    )

    wind_speed = numeric_series(
        df,
        wind_speed_column,
    )

    if wind_angle_column is not None:
        wind_angle = convert_wind_angle_to_degrees(
            df[wind_angle_column],
            wind_angle_column,
        )
    else:
        wind_angle = pd.Series(
            np.nan,
            index=df.index,
            dtype="float64",
        )

    if wind_speed_column is not None:
        mapping["Wind_speed"] = wind_speed_column

    if wind_angle_column is not None:
        mapping["Wind_angle"] = wind_angle_column

    wind_u_column = find_column(
        df,
        "wind_u",
    )

    wind_v_column = find_column(
        df,
        "wind_v",
    )

    wind_w_column = find_column(
        df,
        "wind_w",
    )

    # Derive wind measurements when u and v components exist.
    if (
        wind_u_column is not None
        and wind_v_column is not None
    ):
        wind_u = numeric_series(
            df,
            wind_u_column,
        )

        wind_v = numeric_series(
            df,
            wind_v_column,
        )

        if wind_w_column is not None:
            wind_w = numeric_series(
                df,
                wind_w_column,
            )
        else:
            wind_w = pd.Series(
                0.0,
                index=df.index,
                dtype="float64",
            )

        derived_speed = np.sqrt(
            wind_u ** 2
            + wind_v ** 2
            + wind_w ** 2
        )

        # Direction wind comes FROM.
        derived_angle = (
            np.degrees(
                np.arctan2(
                    -wind_u,
                    -wind_v,
                )
            )
            + 360
        ) % 360

        wind_speed = wind_speed.combine_first(
            derived_speed
        )

        wind_angle = wind_angle.combine_first(
            derived_angle
        )

        if wind_speed_column is None:
            mapping["Wind_speed"] = (
                f"derived from {wind_u_column}, "
                f"{wind_v_column}"
                + (
                    f", {wind_w_column}"
                    if wind_w_column is not None
                    else ""
                )
            )

        if wind_angle_column is None:
            mapping["Wind_angle"] = (
                f"derived from {wind_u_column}, "
                f"{wind_v_column}"
            )

    # Negative wind speed is invalid.
    wind_speed = wind_speed.where(
        wind_speed >= 0
    )

    return wind_speed, wind_angle, mapping


# ============================================================
# COORDINATE-FRAME CHECK
# ============================================================

def infer_coordinate_frame(
    df,
    x_column,
    y_column,
):
    """
    Identify obvious longitude/latitude position data.

    The code preserves the original coordinates rather than
    silently converting them.
    """

    if x_column is None or y_column is None:
        return "unknown"

    if (
        "lon" in x_column
        or "longitude" in x_column
    ) and (
        "lat" in y_column
        or "latitude" in y_column
    ):
        return "geographic_lon_lat_alt"

    x_values = numeric_series(
        df,
        x_column,
    )

    y_values = numeric_series(
        df,
        y_column,
    )

    valid_rows = (
        x_values.notna()
        & y_values.notna()
    )

    if not valid_rows.any():
        return "unknown"

    valid_x = x_values[valid_rows]
    valid_y = y_values[valid_rows]

    # This suggests longitude/latitude, although it does not
    # prove it with certainty.
    looks_like_geographic = (
        valid_x.between(-180, 180).mean() > 0.99
        and valid_y.between(-90, 90).mean() > 0.99
        and valid_x.abs().median() > 20
        and valid_y.abs().median() > 10
    )

    if looks_like_geographic:
        return "likely_geographic_lon_lat_alt"

    return "unknown_or_cartesian"


# ============================================================
# REPORTING
# ============================================================

def print_mapping_report(mapping, source_name, coordinate_frame):
    """
    Print how every standardized column was created.
    """

    print("\n" + "=" * 80)
    print("COLUMN MAPPING")
    print("=" * 80)

    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if column == "Source_dataset":
            source = f'constant value "{source_name}"'

        elif column == "Coordinate_frame":
            source = coordinate_frame

        else:
            source = mapping.get(
                column,
                "NOT FOUND — filled with NaN",
            )

        print(
            f"{column:<20} <- {source}"
        )


def print_completeness_report(cleaned_df):
    """
    Print how many valid values exist in each required column.
    """

    print("\n" + "=" * 80)
    print("COMPLETENESS REPORT")
    print("=" * 80)

    for column in REQUIRED_COLUMNS:
        valid_count = int(
            cleaned_df[column].notna().sum()
        )

        percent_valid = (
            100
            * valid_count
            / max(len(cleaned_df), 1)
        )

        print(
            f"{column:<20} "
            f"{valid_count:>12,} / "
            f"{len(cleaned_df):,} "
            f"({percent_valid:6.2f}%)"
        )

    completely_missing = [
        column
        for column in REQUIRED_COLUMNS
        if cleaned_df[column].notna().sum() == 0
    ]

    if completely_missing:
        print(
            "\nWARNING: These required columns are "
            "completely missing:"
        )

        for column in completely_missing:
            print(f"  - {column}")

    else:
        print(
            "\nAll required columns contain data."
        )


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def standardize_drone_data(
    telemetry_file,
    output_file,
    source_dataset,
    metadata_file=None,
    keep_original_columns=False,
):
    """
    Standardize one drone telemetry CSV.
    """

    telemetry_path = Path(
        telemetry_file
    )

    output_path = Path(
        output_file
    )

    if not telemetry_path.exists():
        raise FileNotFoundError(
            f"Could not find telemetry file:\n"
            f"{telemetry_path.resolve()}"
        )

    print("\nReading telemetry data:")
    print(telemetry_path.resolve())

    raw_telemetry = pd.read_csv(
        telemetry_path,
        low_memory=False,
    )

    print(
        f"Original telemetry shape: "
        f"{raw_telemetry.shape}"
    )

    telemetry_df = normalize_all_headers(
        raw_telemetry
    )

    # Merge optional metadata.
    if metadata_file is not None:
        metadata_path = Path(
            metadata_file
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Could not find metadata file:\n"
                f"{metadata_path.resolve()}"
            )

        print("\nReading metadata:")
        print(metadata_path.resolve())

        raw_metadata = pd.read_csv(
            metadata_path,
            low_memory=False,
        )

        metadata_df = normalize_all_headers(
            raw_metadata
        )

        telemetry_df = merge_metadata(
            telemetry_df,
            metadata_df,
        )

    mapping = {}

    # --------------------------------------------------------
    # Timestamp and elapsed time
    # --------------------------------------------------------

    timestamp, elapsed_seconds, timestamp_mapping = (
        construct_timestamp(telemetry_df)
    )

    mapping.update(
        timestamp_mapping
    )

    # --------------------------------------------------------
    # Position
    # --------------------------------------------------------

    x_column = find_column(
        telemetry_df,
        "x",
    )

    y_column = find_column(
        telemetry_df,
        "y",
    )

    z_column = find_column(
        telemetry_df,
        "z",
    )

    x_values = numeric_series(
        telemetry_df,
        x_column,
    )

    y_values = numeric_series(
        telemetry_df,
        y_column,
    )

    z_values = numeric_series(
        telemetry_df,
        z_column,
    )

    if x_column is not None:
        mapping["X"] = x_column

    if y_column is not None:
        mapping["Y"] = y_column

    if z_column is not None:
        mapping["Z"] = z_column

    # --------------------------------------------------------
    # Roll, pitch, and yaw
    # --------------------------------------------------------

    roll, pitch, yaw, attitude_mapping = (
        get_roll_pitch_yaw(
            telemetry_df
        )
    )

    mapping.update(
        attitude_mapping
    )

    # --------------------------------------------------------
    # Wind
    # --------------------------------------------------------

    wind_speed, wind_angle, wind_mapping = (
        get_wind_data(
            telemetry_df
        )
    )

    mapping.update(
        wind_mapping
    )

    # --------------------------------------------------------
    # Battery
    # --------------------------------------------------------

    battery_voltage_column = find_column(
        telemetry_df,
        "battery_voltage",
    )

    battery_current_column = find_column(
        telemetry_df,
        "battery_current",
    )

    battery_voltage = numeric_series(
        telemetry_df,
        battery_voltage_column,
    )

    battery_current = numeric_series(
        telemetry_df,
        battery_current_column,
    )

    if battery_voltage_column is not None:
        mapping["Battery_V"] = (
            battery_voltage_column
        )

    if battery_current_column is not None:
        mapping["Battery_C"] = (
            battery_current_column
        )

    # A battery voltage of zero or less is not valid.
    battery_voltage = battery_voltage.where(
        battery_voltage > 0
    )

    # Do not automatically remove negative current.
    # Some systems use negative current to represent
    # charging or a different current-direction convention.

    # --------------------------------------------------------
    # Flight ID
    # --------------------------------------------------------

    flight_id_column = find_column(
        telemetry_df,
        "flight_id",
    )

    if flight_id_column is not None:
        flight_id = telemetry_df[
            flight_id_column
        ].copy()

        mapping["Flight_ID"] = (
            flight_id_column
        )

    else:
        flight_id = pd.Series(
            pd.NA,
            index=telemetry_df.index,
            dtype="object",
        )

    # --------------------------------------------------------
    # Coordinate-frame label
    # --------------------------------------------------------

    coordinate_frame = infer_coordinate_frame(
        telemetry_df,
        x_column,
        y_column,
    )

    # --------------------------------------------------------
    # Create standardized dataframe
    # --------------------------------------------------------

    cleaned_df = pd.DataFrame(
        {
            "Timestamp": timestamp,
            "X": x_values,
            "Y": y_values,
            "Z": z_values,
            "Roll": roll,
            "Pitch": pitch,
            "Yaw": yaw,
            "Wind_speed": wind_speed,
            "Wind_angle": wind_angle,
            "Source_dataset": source_dataset,
            "Battery_V": battery_voltage,
            "Battery_C": battery_current,
            "Flight_ID": flight_id,
            "Elapsed_s": elapsed_seconds,
            "Coordinate_frame": coordinate_frame,
        },
        index=telemetry_df.index,
    )

    # Optionally keep all original columns.
    if keep_original_columns:
        original_columns = telemetry_df.add_prefix(
            "Original__"
        )

        cleaned_df = pd.concat(
            [
                cleaned_df,
                original_columns,
            ],
            axis=1,
        )

    # --------------------------------------------------------
    # Remove exact duplicate rows
    # --------------------------------------------------------

    rows_before = len(
        cleaned_df
    )

    cleaned_df = cleaned_df.drop_duplicates().copy()

    duplicates_removed = (
        rows_before
        - len(cleaned_df)
    )

    # --------------------------------------------------------
    # Sort the data
    # --------------------------------------------------------

    sort_columns = []

    if "Flight_ID" in cleaned_df.columns:
        sort_columns.append(
            "Flight_ID"
        )

    if "Timestamp" in cleaned_df.columns:
        sort_columns.append(
            "Timestamp"
        )

    if "Elapsed_s" in cleaned_df.columns:
        sort_columns.append(
            "Elapsed_s"
        )

    if sort_columns:
        cleaned_df = cleaned_df.sort_values(
            by=sort_columns,
            kind="stable",
            na_position="last",
        )

    cleaned_df = cleaned_df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save the cleaned CSV
    # --------------------------------------------------------

    cleaned_df.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S.%f",
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_mapping_report(
        mapping,
        source_dataset,
        coordinate_frame,
    )

    print_completeness_report(
        cleaned_df
    )

    print("\n" + "=" * 80)
    print("CLEANING FINISHED")
    print("=" * 80)

    print(
        f"Rows written: "
        f"{len(cleaned_df):,}"
    )

    print(
        f"Exact duplicate rows removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Cleaned file saved to:\n"
        f"{output_path.resolve()}"
    )

    if coordinate_frame.startswith(
        "likely_geographic"
    ):
        print(
            "\nNOTE: X and Y appear to contain longitude "
            "and latitude rather than distances in meters."
        )

        print(
            "They were preserved instead of being silently "
            "converted."
        )

    return cleaned_df


# ============================================================
# RUN THE CLEANER
# ============================================================

if __name__ == "__main__":

    try:
        cleaned_data = standardize_drone_data(
            telemetry_file=TELEMETRY_FILE,
            metadata_file=METADATA_FILE,
            output_file=OUTPUT_FILE,
            source_dataset=SOURCE_DATASET,
            keep_original_columns=KEEP_ORIGINAL_COLUMNS,
        )

        print("\nFirst five standardized rows:")
        print(
            cleaned_data.head()
        )

    except FileNotFoundError as error:
        print("\nFILE ERROR")
        print(error)

        print(
            "\nMake sure this Python file and both CSV files "
            "are in the same VS Code folder."
        )

    except pd.errors.EmptyDataError:
        print(
            "\nThe CSV file is empty."
        )

    except pd.errors.ParserError as error:
        print(
            "\nPandas could not read one of the CSV files."
        )
        print(error)

    except Exception as error:
        print(
            "\nAn unexpected error occurred:"
        )
        print(
            type(error).__name__,
            error,
        )

