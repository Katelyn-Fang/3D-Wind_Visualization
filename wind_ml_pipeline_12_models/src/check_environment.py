#!/usr/bin/env python3
"""Report whether dependencies for all twelve models can be imported."""
from __future__ import annotations

import importlib
import platform

MODULES = ["numpy", "pandas", "sklearn", "joblib", "torch", "xgboost", "lightgbm", "catboost"]

print(f"Python {platform.python_version()}")
failed = []
for name in MODULES:
    try:
        module = importlib.import_module(name)
        print(f"OK      {name:10s} {getattr(module, '__version__', 'version unavailable')}")
    except Exception as exc:
        failed.append(name)
        print(f"MISSING {name:10s} {exc}")
if failed:
    raise SystemExit("Install missing packages from requirements-all.txt: " + ", ".join(failed))
