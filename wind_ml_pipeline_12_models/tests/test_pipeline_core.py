from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_models import group_endpoint_indices  # noqa: E402
from wind_core import (  # noqa: E402
    add_engineered_features,
    build_direction_targets,
    circular_difference_deg,
    direction_training_weights,
    modeled_to_absolute_angle,
    vectors_to_angle_and_confidence,
    vectors_to_angle_deg,
)


class CoreTests(unittest.TestCase):
    def test_circular_error_wraps(self):
        error = circular_difference_deg(np.array([359.0, 1.0]), np.array([1.0, 359.0]))
        np.testing.assert_allclose(error, np.array([2.0, -2.0]))

    def test_vector_angle_conversion(self):
        angles = vectors_to_angle_deg(np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]))
        np.testing.assert_allclose(angles, np.array([0.0, 90.0, 180.0]))

    def test_direction_confidence_uses_vector_magnitude(self):
        angles, confidence = vectors_to_angle_and_confidence(
            np.array([[1.0, 0.0], [0.1, 0.0]])
        )
        np.testing.assert_allclose(angles, np.array([90.0, 90.0]))
        np.testing.assert_allclose(confidence, np.array([1.0, 0.1]))

    def test_direction_weights_downweight_calm_rows(self):
        speed = np.array([0.1, 0.5, 1.0, 3.0])
        groups = np.array(["a", "a", "a", "a"])
        weights = direction_training_weights(speed, groups, minimum_speed=1.0)
        self.assertLess(weights[0], weights[1])
        self.assertLess(weights[1], weights[2])
        self.assertAlmostEqual(weights[2], weights[3])

    def test_direction_weights_balance_flights(self):
        speed = np.ones(6)
        groups = np.array(["short", "long", "long", "long", "long", "long"])
        weights = direction_training_weights(speed, groups, minimum_speed=1.0)
        self.assertAlmostEqual(weights[groups == "short"].sum(), weights[groups == "long"].sum())

    def test_relative_yaw_target_round_trip(self):
        wind = np.array([10.0, 350.0])
        yaw_rad = np.deg2rad(np.array([350.0, 10.0]))
        vectors, relative, heading = build_direction_targets(
            wind,
            target_mode="relative_yaw",
            yaw=yaw_rad,
            attitude_angle_unit="radians",
            yaw_transform="clockwise_from_north",
        )
        reconstructed = modeled_to_absolute_angle(relative, "relative_yaw", heading)
        np.testing.assert_allclose(reconstructed, wind)
        np.testing.assert_allclose(vectors_to_angle_deg(vectors), relative)

    def test_windows_never_cross_flights(self):
        frame = pd.DataFrame({"_Group_ID": ["a"] * 4 + ["b"] * 5})
        endpoints = group_endpoint_indices(frame, ["a", "b"], 3, use_all_rows=False)
        np.testing.assert_array_equal(endpoints, np.array([2, 3, 6, 7, 8]))
        for endpoint in endpoints:
            group = frame.loc[endpoint, "_Group_ID"]
            self.assertTrue((frame.loc[endpoint - 2 : endpoint, "_Group_ID"] == group).all())

    def test_cartesian_label_vetoes_geographic_inference(self):
        n = 40
        frame = pd.DataFrame({
            # Local-meter coordinates whose medians exceed the old inference
            # thresholds (|X| median > 20, |Y| median > 10).
            "X": np.linspace(0.0, 60.0, n), "Y": np.linspace(0.0, 40.0, n),
            "Z": np.full(n, 12.0),
            "Roll": np.zeros(n), "Pitch": np.zeros(n), "Yaw": np.zeros(n),
            "Battery_V": np.full(n, 16.0), "Battery_C": np.full(n, 5.0),
            "Wind_speed": np.full(n, 2.0), "Wind_angle": np.full(n, 45.0),
            "Flight_ID": ["f"] * n, "Source_dataset": ["s"] * n,
            "Elapsed_s": np.arange(n, dtype=float),
            "Coordinate_frame": ["cartesian_meters"] * n,
        })
        engineered, _, _ = add_engineered_features(frame, attitude_angle_unit="radians")
        self.assertEqual(engineered["Coordinate_is_geographic"].max(), 0.0)
        self.assertAlmostEqual(engineered["Local_east_m"].iloc[-1], 60.0, places=6)
        self.assertAlmostEqual(engineered["Local_north_m"].iloc[-1], 40.0, places=6)

    def test_attitude_angle_auto_detects_degrees(self):
        frame = pd.DataFrame({
            "X": [0.0, 1.0], "Y": [0.0, 1.0], "Z": [10.0, 10.5],
            "Roll": [0.0, 0.0], "Pitch": [0.0, 0.0], "Yaw": [0.0, 90.0],
            "Battery_V": [16.0, 15.9], "Battery_C": [5.0, 5.1],
            "Wind_speed": [1.0, 1.1], "Wind_angle": [0.0, 10.0],
            "Flight_ID": ["f", "f"], "Source_dataset": ["s", "s"],
            "Elapsed_s": [0.0, 1.0],
        })
        engineered, _, metadata = add_engineered_features(frame, attitude_angle_unit="auto")
        self.assertEqual(metadata["attitude_angle_unit_resolved"], "degrees")
        self.assertAlmostEqual(engineered.loc[1, "Yaw_sin"], 1.0, places=6)
        self.assertIn("Yaw_delta_rad", engineered.columns)


if __name__ == "__main__":
    unittest.main()
