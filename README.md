# 3D-Wind_Visualization

- `3DModel/`: trained-ML simulator and held-out ML-versus-physics comparison.
- `3DModel-Physics/`: physics-only simulator. It defaults to the independently
  identified paper-based baseline and also preserves the hypothetical
  inverse-drag demonstration as an optional mode.

The physics baseline and trained ML model are evaluated independently; neither
model consumes the other model's predictions.
