"""PyTorch MLP, LSTM, TCN, and Transformer training for wind telemetry."""
from __future__ import annotations

import copy
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from .wind_core import (
        append_prediction_metrics,
        build_direction_targets,
        direction_training_weights,
        equal_flight_weights,
        modeled_to_absolute_angle,
        prediction_frame,
        vectors_to_angle_and_confidence,
        write_metrics,
    )
except ImportError:  # Support direct imports when src/ itself is on sys.path.
    from wind_core import (
        append_prediction_metrics,
        build_direction_targets,
        direction_training_weights,
        equal_flight_weights,
        modeled_to_absolute_angle,
        prediction_frame,
        vectors_to_angle_and_confidence,
        write_metrics,
    )

NEURAL_MODELS = ["mlp", "lstm", "tcn", "transformer"]


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise ImportError("PyTorch is required for neural models. Install requirements-all.txt") from exc
    return torch, nn, DataLoader, Dataset


@dataclass
class PreprocessorState:
    imputer_statistics: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


class ArrayPreprocessor:
    """Median imputation and standardization with a portable numeric state."""

    def __init__(self) -> None:
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "ArrayPreprocessor":
        X = np.asarray(X, dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        transformed = self.imputer.fit_transform(X)
        self.scaler.fit(transformed)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        transformed = self.imputer.transform(X)
        return self.scaler.transform(transformed).astype(np.float32)

    def state(self) -> PreprocessorState:
        return PreprocessorState(
            imputer_statistics=np.asarray(self.imputer.statistics_, dtype=np.float32),
            scaler_mean=np.asarray(self.scaler.mean_, dtype=np.float32),
            scaler_scale=np.asarray(self.scaler.scale_, dtype=np.float32),
        )


def group_endpoint_indices(
    engineered: pd.DataFrame,
    allowed_groups: Iterable[str],
    sequence_length: int,
    *,
    use_all_rows: bool,
) -> np.ndarray:
    allowed = set(allowed_groups)
    endpoints: List[np.ndarray] = []
    for group_id, part in engineered.groupby("_Group_ID", sort=False, observed=True):
        if group_id not in allowed:
            continue
        indices = part.index.to_numpy(dtype=np.int64)
        if use_all_rows:
            endpoints.append(indices)
        elif len(indices) >= sequence_length:
            endpoints.append(indices[sequence_length - 1 :])
    if not endpoints:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(endpoints)


def split_training_groups(
    groups: List[str], validation_fraction: float, random_seed: int
) -> Tuple[List[str], List[str]]:
    groups = sorted(set(groups))
    if len(groups) < 2:
        raise ValueError("Neural training requires at least two training flights.")
    if len(groups) == 2:
        return [groups[0]], [groups[1]]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_fraction, random_state=random_seed
    )
    representative = np.arange(len(groups))
    fit_pos, val_pos = next(splitter.split(representative, groups=np.asarray(groups)))
    return [groups[i] for i in fit_pos], [groups[i] for i in val_pos]


def endpoint_weights(engineered: pd.DataFrame, endpoints: np.ndarray) -> np.ndarray:
    return equal_flight_weights(engineered.iloc[endpoints]["_Group_ID"])


def build_dataset_class():
    torch, _, _, Dataset = _import_torch()

    class TelemetryDataset(Dataset):
        def __init__(
            self,
            matrix: np.ndarray,
            endpoints: np.ndarray,
            speed: np.ndarray,
            direction: np.ndarray,
            flight_weight: np.ndarray,
            direction_weight: np.ndarray,
            sequence_length: int,
            use_all_rows: bool,
            speed_mean: float,
            speed_std: float,
        ) -> None:
            self.matrix = matrix
            self.endpoints = endpoints.astype(np.int64)
            self.speed = speed
            self.direction = direction
            self.flight_weight = flight_weight.astype(np.float32)
            self.direction_weight = direction_weight.astype(np.float32)
            self.sequence_length = sequence_length
            self.use_all_rows = use_all_rows
            self.speed_mean = float(speed_mean)
            self.speed_std = float(max(speed_std, 1e-6))

        def __len__(self) -> int:
            return len(self.endpoints)

        def __getitem__(self, item: int):
            endpoint = int(self.endpoints[item])
            if self.use_all_rows:
                x = self.matrix[endpoint]
            else:
                x = self.matrix[endpoint - self.sequence_length + 1 : endpoint + 1]
            y_speed = (self.speed[endpoint] - self.speed_mean) / self.speed_std
            y_direction = self.direction[endpoint]
            return (
                torch.from_numpy(np.asarray(x, dtype=np.float32)),
                torch.tensor(y_speed, dtype=torch.float32),
                torch.from_numpy(np.asarray(y_direction, dtype=np.float32)),
                torch.tensor(self.flight_weight[item], dtype=torch.float32),
                torch.tensor(self.direction_weight[item], dtype=torch.float32),
                endpoint,
            )

    return TelemetryDataset


def build_network(model_name: str, input_size: int, args: Any):
    torch, nn, _, _ = _import_torch()

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, args.hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, 3),
            )

        def forward(self, x):
            return self.net(x)

    class LSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                batch_first=True,
                dropout=args.dropout if args.num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(args.hidden_size),
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, 3),
            )

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])

    class CausalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
            super().__init__()
            kernel = 3
            self.left_pad = dilation * (kernel - 1)
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel, dilation=dilation)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel, dilation=dilation)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(args.dropout)
            self.residual = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels else nn.Identity()
            )

        def _causal(self, conv, x):
            return conv(nn.functional.pad(x, (self.left_pad, 0)))

        def forward(self, x):
            residual = self.residual(x)
            x = self.dropout(self.activation(self._causal(self.conv1, x)))
            x = self.dropout(self.activation(self._causal(self.conv2, x)))
            return self.activation(x + residual)

    class TCNNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            blocks = []
            in_channels = input_size
            for layer in range(max(1, args.num_layers)):
                blocks.append(CausalBlock(in_channels, args.hidden_size, dilation=2**layer))
                in_channels = args.hidden_size
            self.tcn = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_size, 3),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            x = self.tcn(x)
            return self.head(x[:, :, -1])

    class TransformerNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            heads = max(1, args.attention_heads)
            d_model = max(heads, args.hidden_size)
            d_model = int(math.ceil(d_model / heads) * heads)
            self.input_projection = nn.Linear(input_size, d_model)
            self.position = nn.Parameter(torch.zeros(1, args.sequence_length, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 2,
                dropout=args.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, args.num_layers))
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))

        def forward(self, x):
            x = self.input_projection(x)
            x = x + self.position[:, : x.shape[1], :]
            x = self.encoder(x)
            return self.head(x[:, -1, :])

    if model_name == "mlp":
        return MLP()
    if model_name == "lstm":
        return LSTMNet()
    if model_name == "tcn":
        return TCNNet()
    if model_name == "transformer":
        return TransformerNet()
    raise ValueError(model_name)


def choose_device(requested: str):
    torch, _, _, _ = _import_torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot see a CUDA GPU.")
    return torch.device(requested)


def set_determinism(seed: int, n_jobs: int) -> None:
    torch, _, _, _ = _import_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, n_jobs))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def weighted_loss(outputs, speed_target, direction_target, flight_weight, direction_weight, args):
    torch, _, _, _ = _import_torch()
    speed_error = torch.nn.functional.smooth_l1_loss(
        outputs[:, 0], speed_target, reduction="none"
    )
    speed_loss = (flight_weight * speed_error).sum() / flight_weight.sum().clamp_min(1e-8)

    predicted_direction = outputs[:, 1:3]
    cosine_similarity = torch.nn.functional.cosine_similarity(
        predicted_direction, direction_target, dim=1, eps=1e-8
    )
    direction_error = 1.0 - cosine_similarity
    direction_loss = (
        direction_weight * direction_error
    ).sum() / direction_weight.sum().clamp_min(1e-8)

    norm_error = (predicted_direction.norm(dim=1) - 1.0) ** 2
    norm_loss = (
        direction_weight * norm_error
    ).sum() / direction_weight.sum().clamp_min(1e-8)

    total_loss = (
        speed_loss
        + args.direction_loss_weight * direction_loss
        + args.direction_norm_weight * norm_loss
    )
    return total_loss, {
        "total_loss": total_loss,
        "speed_loss": speed_loss,
        "direction_loss": direction_loss,
        "direction_norm_loss": norm_loss,
    }


def run_epoch(model, loader, optimizer, device, args, *, training: bool) -> Dict[str, float]:
    torch, _, _, _ = _import_torch()
    model.train(training)
    totals = {
        "total_loss": 0.0,
        "speed_loss": 0.0,
        "direction_loss": 0.0,
        "direction_norm_loss": 0.0,
    }
    total_rows = 0
    for x, y_speed, y_direction, flight_weight, direction_weight, _ in loader:
        x = x.to(device)
        y_speed = y_speed.to(device)
        y_direction = y_direction.to(device)
        flight_weight = flight_weight.to(device)
        direction_weight = direction_weight.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(x)
            loss, parts = weighted_loss(
                outputs, y_speed, y_direction, flight_weight, direction_weight, args
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        for key in totals:
            totals[key] += float(parts[key].detach().cpu()) * len(x)
        total_rows += len(x)
    return {key: value / max(1, total_rows) for key, value in totals.items()}


def make_loader(dataset, args, *, shuffle: bool):
    _, _, DataLoader, _ = _import_torch()
    generator = None
    if shuffle:
        torch, _, _, _ = _import_torch()
        generator = torch.Generator().manual_seed(args.random_seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )


def make_dataset(
    engineered: pd.DataFrame,
    matrix: np.ndarray,
    endpoints: np.ndarray,
    speed: np.ndarray,
    direction: np.ndarray,
    args: Any,
    speed_mean: float,
    speed_std: float,
):
    DatasetClass = build_dataset_class()
    flight_weights = endpoint_weights(engineered, endpoints)
    direction_weights = direction_training_weights(
        speed[endpoints],
        engineered.iloc[endpoints]["_Group_ID"],
        minimum_speed=args.direction_min_speed,
    )
    return DatasetClass(
        matrix,
        endpoints,
        speed,
        direction,
        flight_weights,
        direction_weights,
        args.sequence_length,
        args.model == "mlp",
        speed_mean,
        speed_std,
    )


def tune_epoch_count(
    args: Any,
    engineered: pd.DataFrame,
    raw_matrix: np.ndarray,
    feature_count: int,
    speed: np.ndarray,
    direction: np.ndarray,
    manifest_train_groups: List[str],
    device,
) -> Tuple[int, pd.DataFrame]:
    torch, _, _, _ = _import_torch()
    fit_groups, val_groups = split_training_groups(
        manifest_train_groups, args.validation_fraction, args.random_seed
    )
    fit_rows = engineered.index[engineered["_Group_ID"].isin(fit_groups)].to_numpy()
    preprocessor = ArrayPreprocessor().fit(raw_matrix[fit_rows])
    matrix = preprocessor.transform(raw_matrix)
    fit_endpoints = group_endpoint_indices(
        engineered, fit_groups, args.sequence_length, use_all_rows=args.model == "mlp"
    )
    val_endpoints = group_endpoint_indices(
        engineered, val_groups, args.sequence_length, use_all_rows=args.model == "mlp"
    )
    if not len(fit_endpoints) or not len(val_endpoints):
        raise ValueError("The development sample is too short to create neural train/validation endpoints.")
    speed_mean = float(np.mean(speed[fit_endpoints]))
    speed_std = float(np.std(speed[fit_endpoints])) or 1.0
    fit_dataset = make_dataset(
        engineered, matrix, fit_endpoints, speed, direction, args, speed_mean, speed_std
    )
    val_dataset = make_dataset(
        engineered, matrix, val_endpoints, speed, direction, args, speed_mean, speed_std
    )
    fit_loader = make_loader(fit_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)

    model = build_network(args.model, feature_count, args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 1
    patience_left = args.early_stopping_patience
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, fit_loader, optimizer, device, args, training=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, training=False)
        val_loss = val_metrics["total_loss"]
        rows.append({
            "stage": "validation",
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_speed_loss": train_metrics["speed_loss"],
            "train_direction_loss": train_metrics["direction_loss"],
            "train_direction_norm_loss": train_metrics["direction_norm_loss"],
            "validation_total_loss": val_metrics["total_loss"],
            "validation_speed_loss": val_metrics["speed_loss"],
            "validation_direction_loss": val_metrics["direction_loss"],
            "validation_direction_norm_loss": val_metrics["direction_norm_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if val_loss < best_val - args.early_stopping_min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    # best_state is kept only to verify serialization during tuning; final training starts clean.
    model.load_state_dict(best_state)
    return best_epoch, pd.DataFrame(rows)


def fit_final_model(
    args: Any,
    engineered: pd.DataFrame,
    raw_matrix: np.ndarray,
    feature_count: int,
    speed: np.ndarray,
    direction: np.ndarray,
    train_groups: List[str],
    best_epoch: int,
    device,
):
    torch, _, _, _ = _import_torch()
    train_rows = engineered.index[engineered["_Group_ID"].isin(train_groups)].to_numpy()
    preprocessor = ArrayPreprocessor().fit(raw_matrix[train_rows])
    matrix = preprocessor.transform(raw_matrix)
    train_endpoints = group_endpoint_indices(
        engineered, train_groups, args.sequence_length, use_all_rows=args.model == "mlp"
    )
    speed_mean = float(np.mean(speed[train_endpoints]))
    speed_std = float(np.std(speed[train_endpoints])) or 1.0
    dataset = make_dataset(
        engineered, matrix, train_endpoints, speed, direction, args, speed_mean, speed_std
    )
    loader = make_loader(dataset, args, shuffle=True)
    model = build_network(args.model, feature_count, args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rows = []
    for epoch in range(1, best_epoch + 1):
        train_metrics = run_epoch(model, loader, optimizer, device, args, training=True)
        rows.append({
            "stage": "final_fit",
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_speed_loss": train_metrics["speed_loss"],
            "train_direction_loss": train_metrics["direction_loss"],
            "train_direction_norm_loss": train_metrics["direction_norm_loss"],
            "validation_total_loss": np.nan,
            "validation_speed_loss": np.nan,
            "validation_direction_loss": np.nan,
            "validation_direction_norm_loss": np.nan,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
    return model, preprocessor, matrix, speed_mean, speed_std, train_endpoints, pd.DataFrame(rows)


def predict_model(model, dataset, args, device):
    torch, _, _, _ = _import_torch()
    loader = make_loader(dataset, args, shuffle=False)
    outputs = []
    endpoints = []
    model.eval()
    with torch.no_grad():
        for x, _, _, _, _, endpoint in loader:
            outputs.append(model(x.to(device)).cpu().numpy())
            endpoints.append(np.asarray(endpoint, dtype=np.int64))
    return np.concatenate(outputs), np.concatenate(endpoints)


def gradient_feature_importance(model, dataset, feature_columns: List[str], args: Any, device) -> pd.DataFrame:
    torch, _, _, _ = _import_torch()
    loader = make_loader(dataset, args, shuffle=False)
    totals = np.zeros(len(feature_columns), dtype=float)
    rows = 0
    model.eval()
    for x, _, _, _, _, _ in loader:
        if rows >= args.importance_sample_size:
            break
        remaining = args.importance_sample_size - rows
        x = x[:remaining].to(device)
        x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        output = model(x)
        score = output[:, 0].mean() + output[:, 1:3].abs().mean()
        score.backward()
        grad = x.grad.detach().abs().cpu().numpy()
        if grad.ndim == 3:
            grad = grad.mean(axis=1)
        totals += grad.sum(axis=0)
        rows += len(grad)
    scores = totals / max(rows, 1)
    return pd.DataFrame(
        {
            "method": "mean_absolute_input_gradient",
            "target": "joint_speed_direction",
            "feature": feature_columns,
            "importance_mean": scores,
            "importance_std": np.nan,
        }
    ).sort_values("importance_mean", ascending=False)


def train_neural(
    args: Any,
    engineered: pd.DataFrame,
    feature_columns: List[str],
    feature_metadata: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    start = time.perf_counter()
    torch, _, _, _ = _import_torch()
    set_determinism(args.random_seed, args.n_jobs)
    device = choose_device(args.device)

    raw_matrix = engineered[feature_columns].to_numpy(dtype=float)
    speed = engineered["Wind_speed"].to_numpy(dtype=float)
    angles = engineered["Wind_angle"].to_numpy(dtype=float)
    direction, _, yaw_heading = build_direction_targets(
        angles,
        target_mode=args.direction_target,
        yaw=engineered["Yaw"].to_numpy(dtype=float),
        attitude_angle_unit=feature_metadata["attitude_angle_unit_resolved"],
        yaw_transform=args.yaw_transform,
    )
    direction = direction.astype(np.float32)
    groups = engineered["_Group_ID"]
    train_groups = sorted(set(groups.iloc[train_idx]))
    test_groups = sorted(set(groups.iloc[test_idx]))

    best_epoch, validation_history = tune_epoch_count(
        args,
        engineered,
        raw_matrix,
        len(feature_columns),
        speed,
        direction,
        train_groups,
        device,
    )
    model, preprocessor, matrix, speed_mean, speed_std, train_endpoints, final_history = fit_final_model(
        args,
        engineered,
        raw_matrix,
        len(feature_columns),
        speed,
        direction,
        train_groups,
        best_epoch,
        device,
    )
    history = pd.concat([validation_history, final_history], ignore_index=True)
    history.to_csv(output_dir / "training_history.csv", index=False)

    test_endpoints = group_endpoint_indices(
        engineered,
        test_groups,
        args.sequence_length,
        use_all_rows=args.model == "mlp",
    )
    if not len(test_endpoints):
        raise ValueError("No test endpoints exist for the requested sequence length.")
    test_dataset = make_dataset(
        engineered,
        matrix,
        test_endpoints,
        speed,
        direction,
        args,
        speed_mean,
        speed_std,
    )
    output, predicted_endpoints = predict_model(model, test_dataset, args, device)
    predicted_speed = np.maximum(output[:, 0] * speed_std + speed_mean, 0.0)
    predicted_vectors = output[:, 1:3]
    predicted_modeled_angle, direction_confidence = vectors_to_angle_and_confidence(
        predicted_vectors
    )
    predicted_angle = modeled_to_absolute_angle(
        predicted_modeled_angle,
        args.direction_target,
        None if yaw_heading is None else yaw_heading[predicted_endpoints],
    )
    predictions = prediction_frame(
        engineered,
        predicted_endpoints,
        predicted_speed,
        predicted_angle,
        args.comparison_sequence_length,
    )
    predictions["Predicted_direction_sin"] = predicted_vectors[:, 0]
    predictions["Predicted_direction_cos"] = predicted_vectors[:, 1]
    predictions["Predicted_direction_confidence"] = direction_confidence
    endpoint_speed = speed[predicted_endpoints]
    predictions["Direction_reliability_weight"] = (
        np.clip(endpoint_speed / args.direction_min_speed, 0.05, 1.0)
        if args.direction_min_speed > 0
        else 1.0
    )
    predictions["Direction_target_mode"] = args.direction_target
    predictions["Speed_model_name"] = args.model
    predictions["Direction_model_name"] = args.model
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    metrics: Dict[str, Any] = {
        "script_version": args.script_version,
        "model_name": args.model,
        "model_family": "neural_tabular" if args.model == "mlp" else "neural_sequence",
        "evaluation_scope": "all_test_rows" if args.model == "mlp" else "sequence_endpoints",
        "data_file": str(Path(args.data).resolve()),
        "split_manifest": str(Path(args.split_manifest).resolve()),
        "rows_total": len(engineered),
        "raw_rows_train": len(train_idx),
        "raw_rows_test": len(test_idx),
        "rows_train": len(train_endpoints),
        "rows_test": len(test_endpoints),
        "flights_total": int(groups.nunique()),
        "flights_train": len(train_groups),
        "flights_test": len(test_groups),
        "random_seed": args.random_seed,
        "feature_count": len(feature_columns),
        "time_features_enabled": feature_metadata["time_features_enabled"],
        "attitude_angle_unit_resolved": feature_metadata["attitude_angle_unit_resolved"],
        "comparison_sequence_length": args.comparison_sequence_length,
        "direction_target_mode": args.direction_target,
        "direction_min_speed": args.direction_min_speed,
        "direction_loss_weight": args.direction_loss_weight,
        "direction_norm_weight": args.direction_norm_weight,
        "sequence_length": 1 if args.model == "mlp" else args.sequence_length,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "attention_heads": args.attention_heads if args.model == "transformer" else None,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    metrics, per_flight = append_prediction_metrics(metrics, predictions)
    per_flight.to_csv(output_dir / "per_flight_metrics.csv", index=False)

    pre_state = preprocessor.state()
    checkpoint = {
        "script_version": args.script_version,
        "model_name": args.model,
        "model_state_dict": model.cpu().state_dict(),
        "model_config": {
            "input_size": len(feature_columns),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "attention_heads": args.attention_heads,
            "dropout": args.dropout,
            "sequence_length": args.sequence_length,
            "direction_target_mode": args.direction_target,
            "yaw_transform": args.yaw_transform,
            "direction_loss_weight": args.direction_loss_weight,
            "direction_norm_weight": args.direction_norm_weight,
        },
        "feature_columns": feature_columns,
        "preprocessor": {
            "imputer_statistics": pre_state.imputer_statistics,
            "scaler_mean": pre_state.scaler_mean,
            "scaler_scale": pre_state.scaler_scale,
        },
        "target_scaling": {"speed_mean": speed_mean, "speed_std": speed_std},
        "training_arguments": vars(args),
        "split_groups": {"train": train_groups, "test": test_groups},
    }
    torch.save(checkpoint, output_dir / "wind_model.pt")
    model.to(device)

    if args.feature_importance:
        importance = gradient_feature_importance(
            model, test_dataset, feature_columns, args, device
        )
        importance.to_csv(output_dir / "feature_importance.csv", index=False)

    metrics["training_seconds"] = time.perf_counter() - start
    write_metrics(metrics, output_dir)
    return metrics
