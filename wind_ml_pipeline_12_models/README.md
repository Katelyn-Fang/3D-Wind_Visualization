# BU RISE Wind-Model Pipeline — 12-Model Edition

This package benchmarks twelve wind-speed and wind-direction models with one frozen, flight-level train/test split. It extends the original Extra Trees pipeline without changing its key safeguards: entire flights stay together, engineered features are causal, and wind direction is trained through sine/cosine targets.

Version `0.3.0` adds direction-focused training, confidence diagnostics, hybrid speed/direction tabular models, and separate SCC direction grids. See [`DIRECTION_ACCURACY_UPDATES.md`](DIRECTION_ACCURACY_UPDATES.md) for the exact file-by-file changes.

## Included models

| # | CLI model name | Model | Family | Default final hardware |
|---:|---|---|---|---|
| 1 | `dummy` | Dummy mean baseline | Baseline | CPU |
| 2 | `ridge` | Ridge regression | Linear | CPU |
| 3 | `decision_tree` | Decision tree | Tree | CPU |
| 4 | `random_forest` | Random forest | Bagged trees | CPU |
| 5 | `extra_trees` | Extra Trees | Bagged trees | CPU |
| 6 | `xgboost` | XGBoost | Boosting | CPU first |
| 7 | `lightgbm` | LightGBM | Boosting | CPU first |
| 8 | `catboost` | CatBoost | Boosting | CPU first |
| 9 | `mlp` | PyTorch multilayer perceptron | Tabular neural net | GPU final |
| 10 | `lstm` | LSTM | Sequence neural net | GPU |
| 11 | `tcn` | Temporal convolutional network | Sequence neural net | GPU |
| 12 | `transformer` | Transformer encoder | Sequence neural net | GPU |

## What “tested” means

The included local smoke test verifies that each model can:

1. load the same standardized schema and frozen flight split;
2. train without crossing flight boundaries;
3. predict wind speed plus circular direction;
4. save a model artifact, predictions, metrics, and per-flight metrics;
5. participate in one combined comparison table.

The automatically generated synthetic dataset validates software behavior only. Its scores are not project results. Repeat the workflow on a small sample of your real standardized flights before using the SCC.

## Project layout

```text
wind_ml_pipeline_12_models/
├── configs/
│   ├── model_grid.csv
│   ├── direction_experiments.csv
│   └── local_smoke_grid.csv
├── data/
├── results/
├── scripts/
│   ├── run_all_local.ps1
│   └── run_all_local.sh
├── scc/
│   ├── cpu_model_grid.qsub
│   ├── cpu_direction_grid.qsub
│   ├── gpu_model_grid.qsub
│   ├── gpu_direction_grid.qsub
│   ├── gpu_smoke_test.qsub
│   └── summarize_results.qsub
├── src/
│   ├── __init__.py
│   ├── check_environment.py
│   ├── check_angle_conventions.py
│   ├── check_gpu.py
│   ├── generate_synthetic_data.py
│   ├── make_small_sample.py
│   ├── neural_models.py
│   ├── run_local_smoke_test.py
│   ├── summarize_runs.py
│   ├── tabular_models.py
│   ├── train_wind_model.py
│   └── wind_core.py
├── tests/test_pipeline_core.py
├── DIRECTION_ACCURACY_UPDATES.md
├── requirements-core.txt
└── requirements-all.txt
```

# Part A — Build and test locally

## 1. Create the environment on Windows

From PowerShell in the package root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-all.txt
```

Confirm all dependencies:

```powershell
python src/check_environment.py
```

## 2. Run the unit checks

```powershell
python -m unittest discover -s tests -v
```

These tests verify circular-angle handling and confirm that sequence windows cannot cross flight boundaries.

## 3. Run all twelve models on synthetic data

```powershell
python src/run_local_smoke_test.py --quick
```

Equivalent convenience command:

```powershell
.\scripts\run_all_local.ps1
```

The runner creates a small synthetic dataset when needed and executes every model in a separate Python process. That isolates import or model-specific failures and leaves one log per model under:

```text
results/local_smoke_test/logs/
```

The important completion files are:

```text
results/local_smoke_test/smoke_test_status.csv
results/local_smoke_test/model_comparison.csv
results/local_smoke_test/model_ranking_full_scope.csv
results/local_smoke_test/model_ranking_common_endpoints.csv
results/local_smoke_test/model_ranking_direction.csv
results/local_smoke_test/model_ranking_balanced.csv
```

All twelve rows in `smoke_test_status.csv` should say `PASS`.

## 4. Create a small sample from your real data

Place the full standardized CSV at:

```text
data/DJI_primary_standardized.csv
```

Create a flight-preserving development sample:

```powershell
python src/make_small_sample.py `
  --data data/DJI_primary_standardized.csv `
  --output data/small_sample.csv `
  --manifest data/small_split_manifest.csv `
  --n-flights 10 `
  --max-rows-per-flight 1000 `
  --test-size 0.25 `
  --random-seed 42
```

The row cap retains a contiguous segment inside each selected flight; it does not randomly mix rows.

The trainer defaults to `--attitude-angle-unit auto`. It treats yaw as degrees when its magnitude clearly exceeds a radian range; otherwise it uses radians. Override this with `--attitude-angle-unit degrees` or `radians` after confirming your standardized-data units.

The final SCC grid explicitly uses `radians`, matching the standardized dataset used for this project. Change that grid field only when your final CSV uses different attitude units.

## 5. Run all twelve models on the real small sample

```powershell
python src/run_local_smoke_test.py `
  --data data/small_sample.csv `
  --split-manifest data/small_split_manifest.csv `
  --results-dir results/real_small_sample `
  --quick
```

This is still a software/debug run. After it passes, increase neural epochs and tree counts for a more meaningful local comparison:

```powershell
python src/run_local_smoke_test.py `
  --data data/small_sample.csv `
  --split-manifest data/small_split_manifest.csv `
  --results-dir results/real_small_sample_extended
```

## 6. Run one model manually

Extra Trees:

```powershell
python src/train_wind_model.py `
  --data data/small_sample.csv `
  --split-manifest data/small_split_manifest.csv `
  --output-dir results/manual/extra_trees `
  --model extra_trees `
  --n-estimators 100 `
  --min-samples-leaf 10 `
  --max-depth 20 `
  --max-features 0.80 `
  --n-jobs 4 `
  --comparison-sequence-length 30 `
  --feature-importance
```

LSTM on CPU for debugging:

```powershell
python src/train_wind_model.py `
  --data data/small_sample.csv `
  --split-manifest data/small_split_manifest.csv `
  --output-dir results/manual/lstm `
  --model lstm `
  --sequence-length 30 `
  --comparison-sequence-length 30 `
  --hidden-size 64 `
  --num-layers 2 `
  --epochs 10 `
  --batch-size 128 `
  --device cpu `
  --n-jobs 4 `
  --feature-importance
```

# Part B — Fair comparison design

## Fixed flight split

Every model uses the same `small_split_manifest.csv`. No flight can appear in both train and test.

## Full-scope versus common-endpoint metrics

The tabular models and MLP can predict every test row. LSTM, TCN, and Transformer require a complete history window, so they cannot predict the first `sequence_length - 1` rows of a flight.

The pipeline therefore reports:

- normal metrics over each model’s natural evaluation scope;
- `common_*` metrics restricted to rows with enough history for the configured comparison sequence length.

Use `model_ranking_common_endpoints.csv` for direct comparisons across all twelve models. Use the full-scope table to understand each model’s practical coverage.

## Neural validation

Neural models split the manifest’s training flights into temporary fit/validation flights for early stopping. After selecting the epoch count, they restart from a fresh model and train on all manifest training flights for that many epochs. Test flights remain untouched.

# Part C — Output contract

Every run writes:

```text
test_predictions.csv
model_metrics.csv
metrics.json
per_flight_metrics.csv
```

`test_predictions.csv` also includes the predicted sine/cosine components, their vector-magnitude confidence, the direction target mode, and the speed/direction model-family names.

Tabular models write:

```text
wind_model.joblib
```

PyTorch models write:

```text
wind_model.pt
training_history.csv
```

With `--feature-importance`, all model families also write:

```text
feature_importance.csv
```

For tabular models this is permutation importance. For neural models it is mean absolute input-gradient importance, averaged over sequence time steps where applicable.

# Part D — GitHub handoff

Generated CSV data, results, scheduler logs, and model artifacts are ignored by Git.

```powershell
git status
git add README.md DIRECTION_ACCURACY_UPDATES.md .gitignore requirements*.txt src tests scripts scc configs data/README.md results/README.md
git commit -m "Add twelve-model wind benchmark pipeline"
git push origin YOUR_BRANCH
```

# Part E — SCC handoff after local tests pass

## 1. Verify a GPU allocation

Edit `PROJECT_DIR` in `scc/gpu_smoke_test.qsub`, then submit:

```bash
qsub scc/gpu_smoke_test.qsub
```

## 2. Prepare full-data names

The supplied SCC scripts expect:

```text
data/full_standardized.csv
data/full_split_manifest.csv
```

Either use those names or edit both qsub files.

## 3. Submit CPU models

Edit `PROJECT_DIR` in `scc/cpu_model_grid.qsub`:

```bash
qsub scc/cpu_model_grid.qsub
```

This submits eight independent array tasks: the baseline, linear/tree models, and three boosted-tree models.

## 4. Submit neural models

Edit `PROJECT_DIR` in `scc/gpu_model_grid.qsub`:

```bash
qsub scc/gpu_model_grid.qsub
```

This submits four GPU array tasks: MLP, LSTM, TCN, and Transformer. Do not manually set `CUDA_VISIBLE_DEVICES`.

## 5. Summarize after both arrays finish

```bash
qsub scc/summarize_results.qsub
```

Or run interactively:

```bash
python src/summarize_runs.py --results-dir results/scc
```

## 6. Submit direction-focused experiments

The baseline arrays should run first. Then submit the additional direction experiments:

```bash
qsub scc/cpu_direction_grid.qsub
qsub scc/gpu_direction_grid.qsub
```

These jobs compare leaf sizes, hybrid Extra Trees/Random Forest and Extra Trees/CatBoost models, and sequence lengths of 30 versus 60 for TCN and Transformer. Both direction grids pin `--comparison-sequence-length` to 60 (the longest sequence in the grid) so every run's `common_*` metrics cover the identical endpoint subset; the hybrid rows also reuse the `et_leaf3` tree hyperparameters so their speed models match the baseline exactly.

Relative-to-yaw rows are disabled by default. First run:

```bash
python src/check_angle_conventions.py --data data/full_standardized.csv --attitude-angle-unit radians
```

Only enable a relative-yaw row when one yaw transform is clearly supported by both the diagnostic and the dataset documentation.

# Local pass criteria before SCC

Do not submit the full grid until:

1. `python src/check_environment.py` passes;
2. all unit tests pass;
3. all twelve synthetic smoke-test rows say `PASS`;
4. all twelve real-small-sample rows say `PASS`;
5. every run has the expected model artifact and prediction/metrics files;
6. the models report identical train/test flight counts;
7. common-endpoint rankings contain all twelve models;
8. `model_ranking_direction.csv` is created and includes the models with valid ≥1 m/s direction metrics;
9. `test_predictions.csv` contains `Predicted_direction_confidence` and the model-name columns;
10. one `test_predictions.csv` has been loaded successfully by the local Plotly visualizer.
