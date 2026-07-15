# Twelve-Model Smoke-Test Report — v0.3.0

## Result

All twelve model code paths completed successfully on the included synthetic, flight-grouped telemetry sample after the direction-accuracy update.

- Eight tabular models wrote `wind_model.joblib`.
- Four PyTorch models wrote `wind_model.pt` and `training_history.csv`.
- Every run wrote predictions, metrics, and per-flight metrics.
- Every `test_predictions.csv` contained direction sine/cosine outputs, vector-magnitude confidence, target mode, and speed/direction model names.
- The shared summarizer produced full-scope, common-endpoint, direction-focused, and balanced rankings.
- The Extra Trees speed + Random Forest direction hybrid path completed separately.
- The optional relative-to-yaw code path completed on synthetic data, but remains disabled in the SCC grid until the real-data yaw convention is verified.

## Models verified

1. Dummy baseline
2. Ridge regression
3. Decision tree
4. Random forest
5. Extra Trees
6. XGBoost
7. LightGBM
8. CatBoost
9. PyTorch MLP
10. LSTM
11. Temporal convolutional network
12. Transformer encoder

## Unit tests

Eight unit tests passed. They cover:

- circular angle wrapping;
- sine/cosine vector conversion;
- vector-magnitude direction confidence;
- calm-wind direction down-weighting;
- equal total weight across flights;
- relative-yaw target reconstruction;
- sequence windows staying inside flights;
- attitude-angle unit detection and angular feature creation.

## Important interpretation

The included scores are not research findings. The synthetic dataset is intentionally small and learnable, and neural smoke tests use one epoch. The purpose is to verify software behavior before running the same package on real drone flights and then on the BU SCC.

See `SMOKE_TEST_STATUS.csv`, `SMOKE_TEST_MODEL_COMPARISON.csv`, and `DIRECTION_ACCURACY_UPDATES.md`.
