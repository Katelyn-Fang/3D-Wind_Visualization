"""Training for the eight non-neural tabular models."""
from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from .wind_core import (
        append_prediction_metrics,
        build_direction_targets,
        circular_mean_deg,
        direction_training_weights,
        equal_flight_weights,
        evaluate_predictions,
        modeled_to_absolute_angle,
        prediction_frame,
        vectors_to_angle_and_confidence,
        vectors_to_angle_deg,
        write_metrics,
    )
except ImportError:  # Support direct imports when src/ itself is on sys.path.
    from wind_core import (
        append_prediction_metrics,
        build_direction_targets,
        circular_mean_deg,
        direction_training_weights,
        equal_flight_weights,
        evaluate_predictions,
        modeled_to_absolute_angle,
        prediction_frame,
        vectors_to_angle_and_confidence,
        vectors_to_angle_deg,
        write_metrics,
    )



class SklearnRegressorAdapter(RegressorMixin, BaseEstimator):
    """Add current sklearn tags/fitted-state behavior around third-party regressors."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        self.estimator_ = clone(self.estimator)
        if sample_weight is None:
            self.estimator_.fit(X, y)
        else:
            self.estimator_.fit(X, y, sample_weight=sample_weight)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class IndependentMultiOutputRegressor(RegressorMixin, BaseEstimator):
    """Fit one cloned regressor per output without relying on sklearn's estimator tags."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y)
        if y.ndim != 2:
            raise ValueError(f"Expected a two-dimensional target; got shape {y.shape}.")
        self.estimators_ = []
        for column in range(y.shape[1]):
            estimator = clone(self.estimator)
            if sample_weight is None:
                estimator.fit(X, y[:, column])
            else:
                estimator.fit(X, y[:, column], sample_weight=sample_weight)
            self.estimators_.append(estimator)
        return self

    def predict(self, X):
        return np.column_stack([estimator.predict(X) for estimator in self.estimators_])


TABULAR_MODELS = [
    "dummy",
    "ridge",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
]


def _optional_estimator(model_name: str, args: Any) -> Any:
    max_depth = None if args.max_depth == 0 else args.max_depth
    if model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for --model xgboost. Install requirements-all.txt") from exc
        return XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=max_depth or 6,
            learning_rate=args.learning_rate,
            min_child_weight=max(1.0, float(args.min_samples_leaf)),
            subsample=args.subsample,
            colsample_bytree=args.max_features,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=args.n_jobs,
            random_state=args.random_seed,
            verbosity=0,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is required for --model lightgbm. Install requirements-all.txt") from exc
        return LGBMRegressor(
            n_estimators=args.n_estimators,
            max_depth=-1 if max_depth is None else max_depth,
            learning_rate=args.learning_rate,
            min_child_samples=args.min_samples_leaf,
            subsample=args.subsample,
            colsample_bytree=args.max_features,
            n_jobs=args.n_jobs,
            random_state=args.random_seed,
            verbosity=-1,
        )
    if model_name == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("catboost is required for --model catboost. Install requirements-all.txt") from exc
        depth = 8 if max_depth is None else min(max_depth, 10)
        return SklearnRegressorAdapter(CatBoostRegressor(
            iterations=args.n_estimators,
            depth=depth,
            learning_rate=args.learning_rate,
            l2_leaf_reg=3.0,
            random_seed=args.random_seed,
            thread_count=args.n_jobs,
            verbose=False,
            allow_writing_files=False,
            loss_function="RMSE",
        ))
    raise ValueError(model_name)


def build_estimator(model_name: str, args: Any) -> Tuple[Any, bool]:
    """Return a base regressor and whether standardized features are helpful."""
    max_depth = None if args.max_depth == 0 else args.max_depth
    if model_name == "dummy":
        return DummyRegressor(strategy="mean"), False
    if model_name == "ridge":
        return Ridge(alpha=args.ridge_alpha), True
    if model_name == "decision_tree":
        return DecisionTreeRegressor(
            min_samples_leaf=args.min_samples_leaf,
            max_depth=max_depth,
            max_features=args.max_features,
            random_state=args.random_seed,
        ), False
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=max_depth,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
            random_state=args.random_seed,
        ), False
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=max_depth,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
            random_state=args.random_seed,
        ), False
    if model_name in {"xgboost", "lightgbm", "catboost"}:
        return _optional_estimator(model_name, args), False
    raise ValueError(f"Unsupported tabular model: {model_name}")


def build_pipeline(model_name: str, args: Any, *, multi_output: bool) -> Pipeline:
    estimator, scale = build_estimator(model_name, args)
    if multi_output:
        estimator = IndependentMultiOutputRegressor(estimator)
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def direction_neg_circular_mae(estimator: Any, X: pd.DataFrame, y: np.ndarray) -> float:
    actual = vectors_to_angle_deg(y)
    predicted = vectors_to_angle_deg(estimator.predict(X))
    return -float(np.mean(np.abs((predicted - actual + 180.0) % 360.0 - 180.0)))


def _fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Pipeline:
    pipeline.fit(X, y, model__sample_weight=weights)
    return pipeline


def train_tabular(
    args: Any,
    engineered: pd.DataFrame,
    feature_columns: list[str],
    feature_metadata: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    start = time.perf_counter()
    X = engineered[feature_columns]
    speed = engineered["Wind_speed"].to_numpy(dtype=float)
    angles = engineered["Wind_angle"].to_numpy(dtype=float)
    direction, _, yaw_heading = build_direction_targets(
        angles,
        target_mode=args.direction_target,
        yaw=engineered["Yaw"].to_numpy(dtype=float),
        attitude_angle_unit=feature_metadata["attitude_angle_unit_resolved"],
        yaw_transform=args.yaw_transform,
    )
    groups = engineered["_Group_ID"]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    speed_train, speed_test = speed[train_idx], speed[test_idx]
    direction_train = direction[train_idx]
    angle_test = angles[test_idx]
    flight_weights = equal_flight_weights(groups.iloc[train_idx])
    direction_weights = direction_training_weights(
        speed_train,
        groups.iloc[train_idx],
        minimum_speed=args.direction_min_speed,
    )

    direction_model_name = args.direction_model or args.model
    speed_model = build_pipeline(args.model, args, multi_output=False)
    direction_model = build_pipeline(direction_model_name, args, multi_output=True)
    speed_model = _fit_pipeline(speed_model, X_train, speed_train, flight_weights)
    direction_model = _fit_pipeline(direction_model, X_train, direction_train, direction_weights)

    predicted_speed = np.maximum(np.asarray(speed_model.predict(X_test), dtype=float), 0.0)
    predicted_vectors = np.asarray(direction_model.predict(X_test), dtype=float)
    predicted_modeled_angle, direction_confidence = vectors_to_angle_and_confidence(
        predicted_vectors
    )
    predicted_angle = modeled_to_absolute_angle(
        predicted_modeled_angle,
        args.direction_target,
        None if yaw_heading is None else yaw_heading[test_idx],
    )
    predictions = prediction_frame(
        engineered,
        test_idx,
        predicted_speed,
        predicted_angle,
        args.comparison_sequence_length,
    )
    predictions["Predicted_direction_sin"] = predicted_vectors[:, 0]
    predictions["Predicted_direction_cos"] = predicted_vectors[:, 1]
    predictions["Predicted_direction_confidence"] = direction_confidence
    predictions["Direction_reliability_weight"] = (
        np.clip(speed_test / args.direction_min_speed, 0.05, 1.0)
        if args.direction_min_speed > 0
        else 1.0
    )
    predictions["Direction_target_mode"] = args.direction_target
    predictions["Speed_model_name"] = args.model
    predictions["Direction_model_name"] = direction_model_name
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics: Dict[str, Any] = {
        "script_version": args.script_version,
        "model_name": (
            args.model
            if direction_model_name == args.model
            else f"{args.model}+{direction_model_name}_direction"
        ),
        "speed_model_name": args.model,
        "direction_model_name": direction_model_name,
        "model_family": "tabular",
        "evaluation_scope": "all_test_rows",
        "data_file": str(Path(args.data).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "rows_total": len(engineered),
        "rows_train": len(train_idx),
        "rows_test": len(test_idx),
        "flights_total": int(groups.nunique()),
        "flights_train": int(groups.iloc[train_idx].nunique()),
        "flights_test": int(groups.iloc[test_idx].nunique()),
        "random_seed": args.random_seed,
        "feature_count": len(feature_columns),
        "time_features_enabled": feature_metadata["time_features_enabled"],
        "attitude_angle_unit_resolved": feature_metadata["attitude_angle_unit_resolved"],
        "comparison_sequence_length": args.comparison_sequence_length,
        "direction_target_mode": args.direction_target,
        "direction_min_speed": args.direction_min_speed,
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "max_depth": None if args.max_depth == 0 else args.max_depth,
        "max_features": args.max_features,
        "learning_rate": args.learning_rate,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }
    metrics, per_flight = append_prediction_metrics(metrics, predictions)
    per_flight.to_csv(output_dir / "per_flight_metrics.csv", index=False)

    baseline_speed = float(np.average(speed_train, weights=flight_weights))
    baseline_angle = circular_mean_deg(angles[train_idx])
    baseline = evaluate_predictions(
        speed_test,
        np.full_like(speed_test, baseline_speed),
        angle_test,
        np.full_like(angle_test, baseline_angle),
    )
    metrics["baseline_speed_mae_mps"] = baseline["speed_mae_mps"]
    metrics["baseline_direction_all_circular_mae_deg"] = baseline[
        "direction_all_circular_mae_deg"
    ]
    metrics["speed_mae_improvement_vs_constant_fraction"] = (
        (baseline["speed_mae_mps"] - metrics["speed_mae_mps"]) / baseline["speed_mae_mps"]
        if baseline["speed_mae_mps"] > 0 else None
    )

    artifact = {
        "script_version": args.script_version,
        "model_name": metrics["model_name"],
        "speed_model_name": args.model,
        "direction_model_name": direction_model_name,
        "speed_model": speed_model,
        "direction_model": direction_model,
        "feature_columns": feature_columns,
        "feature_metadata": feature_metadata,
        "training_arguments": vars(args),
        "split_groups": {
            "train": sorted(set(groups.iloc[train_idx])),
            "test": sorted(set(groups.iloc[test_idx])),
        },
        "notes": (
            "Direction is trained as independent sine and cosine regressions; "
            "relative_yaw is used only when explicitly requested with a verified transform."
        ),
    }
    joblib.dump(artifact, output_dir / "wind_model.joblib")

    if args.feature_importance and args.model != "dummy":
        sample_n = min(args.importance_sample_size, len(X_test))
        rng = np.random.default_rng(args.random_seed)
        positions = rng.choice(len(X_test), size=sample_n, replace=False)
        X_sample = X_test.iloc[positions]
        speed_sample = speed_test[positions]
        direction_sample = direction[test_idx][positions]
        speed_result = permutation_importance(
            speed_model,
            X_sample,
            speed_sample,
            n_repeats=args.importance_repeats,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            scoring="neg_mean_absolute_error",
        )
        direction_result = permutation_importance(
            direction_model,
            X_sample,
            direction_sample,
            n_repeats=args.importance_repeats,
            random_state=args.random_seed,
            n_jobs=args.n_jobs,
            scoring=direction_neg_circular_mae,
        )
        importance = pd.concat(
            [
                pd.DataFrame({
                    "method": "permutation",
                    "target": "wind_speed",
                    "feature": feature_columns,
                    "importance_mean": speed_result.importances_mean,
                    "importance_std": speed_result.importances_std,
                }),
                pd.DataFrame({
                    "method": "permutation",
                    "target": "wind_direction",
                    "feature": feature_columns,
                    "importance_mean": direction_result.importances_mean,
                    "importance_std": direction_result.importances_std,
                }),
            ],
            ignore_index=True,
        ).sort_values(["target", "importance_mean"], ascending=[True, False])
        importance.to_csv(output_dir / "feature_importance.csv", index=False)

    metrics["training_seconds"] = time.perf_counter() - start
    write_metrics(metrics, output_dir)
    return metrics
