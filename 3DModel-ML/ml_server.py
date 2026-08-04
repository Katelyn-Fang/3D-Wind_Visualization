"""Uvicorn entry point for the repository's canonical 3DModel ML server.

Keeping this file as a thin adapter ensures that 3DModel and 3DModel-ML use
the same artifact loader, telemetry feature engineering, direction convention,
and prediction endpoints. The canonical implementation lives in
``3DModel/ml_server.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


CANONICAL_SERVER_PATH = Path(__file__).resolve().parent.parent / "3DModel" / "ml_server.py"
MODULE_NAME = "repository_3dmodel_ml_server"

spec = importlib.util.spec_from_file_location(MODULE_NAME, CANONICAL_SERVER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load canonical ML server: {CANONICAL_SERVER_PATH}")

canonical_server = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = canonical_server
spec.loader.exec_module(canonical_server)

app = canonical_server.app

__all__ = ["app"]
