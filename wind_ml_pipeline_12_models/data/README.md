Place standardized telemetry CSV files here. The local smoke test automatically creates `synthetic_smoke.csv` and `synthetic_split_manifest.csv` when they are absent. Synthetic data is for software validation only.

The SCC grids expect:

```text
full_standardized.csv
full_split_manifest.csv
```

For the BU RISE standardized data, Roll/Pitch/Yaw are expected in radians and `Wind_angle` in degrees from 0 to 360. Change `attitude_angle_unit` in the config grids only when the final standardized CSV uses a different attitude unit.

## Drone onboard multi-modal sensor dataset

`Clean.py` defaults to the file `Drone onboard multi-modal sensor dataset for complex outdoor scenarios.csv` and writes `drone_onboard_multimodal_cleaned.csv`.

Dataset-specific conversions:

- semicolon delimiter is detected automatically;
- battery millivolts/milliamperes are converted to volts/amperes;
- source `position_x` (latitude) and `position_y` (longitude) are swapped into the project convention `X=longitude`, `Y=latitude`;
- quaternion orientation supplies roll and pitch; the source yaw is retained in radians;
- exact duplicate standardized rows are removed, but missing wind-speed labels are retained in the cleaned artifact and excluded during evaluation.

Run from the project root:

```powershell
python data/Clean.py --input "data/Drone onboard multi-modal sensor dataset for complex outdoor scenarios.csv" --output data/drone_onboard_multimodal_cleaned.csv
```

Use `notebooks/compare_pretrained_models_drone_onboard.ipynb` to evaluate saved baseline models without retraining.
