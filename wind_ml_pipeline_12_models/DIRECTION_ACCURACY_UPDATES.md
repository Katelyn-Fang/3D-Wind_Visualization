# Direction-Accuracy Update Guide (v0.3.0)

This guide applies to the files in this exact ZIP. The package does **not** contain `Clean.py`; it expects an already-standardized CSV. The baseline SCC runs continue to predict absolute wind direction with sine/cosine targets. Relative-to-yaw prediction is available only as an optional experiment after yaw conventions are checked.

## What was already present before this update

The previous package already had:

- absolute direction represented as sine and cosine;
- circular direction metrics at all speeds and at 0.5, 1.0, and 2.0 m/s;
- equal-flight sample weighting;
- smooth down-weighting of low-speed direction rows;
- separate speed and direction estimators for tabular models (using the same family);
- flight-level train/test splits and flight-level neural validation;
- early stopping for neural networks;
- common-endpoint metrics for fair tabular/sequence comparison.

The update builds on those features rather than replacing them.

## File-by-file changes

### `src/__init__.py`

Added a package marker so newly trained `wind_model.joblib` artifacts can be loaded from the project root without manually adding `src/` to `sys.path`. `train_wind_model.py` now imports the training modules through the `src` package, making custom estimator class paths stable inside saved artifacts.

### `src/wind_core.py`

Added reusable direction utilities:

- `direction_training_weights(...)` combines equal-flight weighting with low-speed reliability weighting;
- `vectors_to_angle_and_confidence(...)` returns both the angle and the sine/cosine vector magnitude;
- `build_direction_targets(...)` supports `absolute` and optional `relative_yaw` targets;
- `yaw_to_heading_deg(...)` supports four explicit yaw-reference transforms;
- `modeled_to_absolute_angle(...)` converts relative predictions back to absolute wind direction.

Expanded causal features with:

- wrapped yaw change;
- roll and pitch change in radians;
- yaw, roll, and pitch rates when timestamps are usable;
- horizontal speed;
- course sine/cosine;
- trailing 5-, 15-, and 30-row means and standard deviations.

Expanded per-flight and common-endpoint metrics to include flight-balanced direction performance above 0.5, 1.0, and 2.0 m/s.

### `src/tabular_models.py`

Added support for different model families for speed and direction:

```text
--model extra_trees --direction-model random_forest
```

The speed estimator still predicts wind speed. The direction estimator independently predicts sine and cosine. The saved `wind_model.joblib` now records both model-family names.

`test_predictions.csv` now includes:

- `Predicted_direction_sin`;
- `Predicted_direction_cos`;
- `Predicted_direction_confidence`;
- `Direction_reliability_weight`;
- `Direction_target_mode`;
- `Speed_model_name`;
- `Direction_model_name`.

### `src/neural_models.py`

Replaced ordinary sine/cosine MSE with a direction-aware objective:

- flight-weighted Smooth L1 speed loss;
- speed-weighted circular cosine direction loss;
- a small penalty encouraging direction-vector magnitude near one.

`training_history.csv` now separates total, speed, direction, and direction-norm losses for training and validation.

Neural predictions also include direction components and confidence.

### `src/train_wind_model.py`

Version increased to `0.3.0`.

New or changed arguments:

```text
--direction-model
--direction-target absolute|relative_yaw
--yaw-transform
--direction-min-speed         default 1.0
--direction-loss-weight       default 2.0
--direction-norm-weight       default 0.05
--epochs                      default 60
--early-stopping-patience     default 8
```

`relative_yaw` is rejected unless a yaw transform is supplied.

### `src/check_angle_conventions.py`

New diagnostic script. It compares common yaw transforms with GPS/course direction on moving rows:

```powershell
python src/check_angle_conventions.py `
  --data data/small_sample.csv `
  --attitude-angle-unit radians `
  --output results/yaw_convention_check.csv
```

This is only a diagnostic because aircraft heading and GPS course can differ during hover, turns, or side-slip. Do not enable relative-yaw SCC rows unless the result is clear and agrees with the dataset documentation.

### `src/summarize_runs.py`

In addition to the existing comparison files, it now writes:

```text
model_ranking_direction.csv
model_ranking_balanced.csv
```

The direction ranking uses flight-balanced common-endpoint circular MAE for rows with true wind speed at least 1.0 m/s. The balanced ranking averages speed rank and direction rank; it does not average m/s and degrees directly.

### `src/run_local_smoke_test.py`

Added command-line forwarding for the direction target/loss settings and output-schema validation. A run is marked failed if the new direction-confidence/model columns are missing.

### `src/generate_synthetic_data.py`

Synthetic data now includes:

- direction values crossing the 0°/360° boundary;
- calm intervals below 1 m/s;
- stronger-wind intervals.

This improves software tests for circular targets and low-speed weighting.

### `tests/test_pipeline_core.py`

Added tests for:

- direction confidence;
- calm-wind down-weighting;
- equal total weight per flight;
- relative-yaw target reconstruction;
- new angular features.

### `configs/model_grid.csv`

The twelve-model baseline grid now uses:

- 500 trees for Extra Trees and Random Forest;
- 800 rounds for XGBoost and LightGBM;
- 1,000 CatBoost iterations;
- 60 maximum neural epochs;
- hidden size 128;
- direction minimum speed 1.0 m/s;
- direction loss weight 2.0;
- direction norm weight 0.05;
- explicit `radians` attitude units for the standardized SCC dataset.

Change `attitude_angle_unit` only if your final standardized CSV is not in radians.

### `configs/direction_experiments.csv`

New direction-focused grid containing:

- Extra Trees leaf-size comparisons;
- Random Forest leaf-size comparisons;
- Extra Trees speed + Random Forest direction;
- Extra Trees speed + CatBoost direction;
- TCN sequence lengths 30 and 60;
- Transformer sequence lengths 30 and 60.

Relative-yaw rows are present but have `enabled=0` and an invalid placeholder transform. They cannot be submitted accidentally by the supplied qsub scripts.

### SCC scripts

Updated baseline scripts:

```text
scc/cpu_model_grid.qsub
scc/gpu_model_grid.qsub
```

Added direction experiment scripts:

```text
scc/cpu_direction_grid.qsub
scc/gpu_direction_grid.qsub
```

Updated summary script:

```text
scc/summarize_results.qsub
```

All SCC scripts still require editing `PROJECT_DIR`. They use `NSLOTS` rather than `-1` for CPU parallelism.

## Local validation order

From the package root:

```powershell
python -m unittest discover -s tests -v
python src/run_local_smoke_test.py --quick
```

Then test the current real-data sample:

```powershell
python src/run_local_smoke_test.py `
  --data data/small_sample.csv `
  --split-manifest data/small_split_manifest.csv `
  --results-dir results/real_small_direction_update `
  --attitude-angle-unit radians
```

Test the first hybrid candidate:

```powershell
python src/train_wind_model.py `
  --data data/small_sample.csv `
  --split-manifest data/small_split_manifest.csv `
  --output-dir results/manual/et_rf_hybrid `
  --model extra_trees `
  --direction-model random_forest `
  --n-estimators 300 `
  --min-samples-leaf 3 `
  --max-depth 20 `
  --max-features 0.8 `
  --direction-min-speed 1.0 `
  --direction-target absolute `
  --attitude-angle-unit radians `
  --n-jobs 4
```

## SCC submission order

1. Edit `PROJECT_DIR` in every qsub file.
2. Confirm the full data and split manifest names.
3. Submit `scc/gpu_smoke_test.qsub`.
4. Submit the baseline arrays:

```bash
qsub scc/cpu_model_grid.qsub
qsub scc/gpu_model_grid.qsub
```

5. Submit the direction-focused arrays:

```bash
qsub scc/cpu_direction_grid.qsub
qsub scc/gpu_direction_grid.qsub
```

6. After all jobs finish:

```bash
qsub scc/summarize_results.qsub
```

Use `model_ranking_direction.csv` for direction selection and `model_ranking_common_endpoints.csv` for the original broad benchmark.
