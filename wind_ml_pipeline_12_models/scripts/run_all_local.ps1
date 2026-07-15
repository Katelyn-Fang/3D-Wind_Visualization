$ErrorActionPreference = "Stop"
python src/check_environment.py
python src/run_local_smoke_test.py --quick
