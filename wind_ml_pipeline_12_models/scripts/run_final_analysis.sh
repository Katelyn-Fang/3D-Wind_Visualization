#!/bin/bash
set -euo pipefail

RESULTS_DIR="${1:-results/scc/baseline}"
OUTPUT_DIR="${2:-results/scc/final_analysis}"
EXPECTED_MODELS="${EXPECTED_MODELS:-12}"
DIRECTION_MIN_SPEED="${DIRECTION_MIN_SPEED:-1.0}"

export MPLBACKEND=Agg

python src/select_final_models.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --expected-models "$EXPECTED_MODELS" \
  --direction-min-speed "$DIRECTION_MIN_SPEED"

python src/make_final_project_figures.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --expected-models "$EXPECTED_MODELS" \
  --direction-min-speed "$DIRECTION_MIN_SPEED" \
  --formats png,pdf

echo
echo "Final analysis complete."
echo "Selection: $OUTPUT_DIR/final_model_selection.txt"
echo "Figures:   $OUTPUT_DIR/figures/"
