Place standardized telemetry CSV files here. The local smoke test automatically creates `synthetic_smoke.csv` and `synthetic_split_manifest.csv` when they are absent. Synthetic data is for software validation only.

The SCC grids expect:

```text
full_standardized.csv
full_split_manifest.csv
```

For the BU RISE standardized data, Roll/Pitch/Yaw are expected in radians and `Wind_angle` in degrees from 0 to 360. Change `attitude_angle_unit` in the config grids only when the final standardized CSV uses a different attitude unit.
