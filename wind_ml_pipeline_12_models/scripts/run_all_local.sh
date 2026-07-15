#!/usr/bin/env bash
set -euo pipefail
python src/check_environment.py
python src/run_local_smoke_test.py --quick
