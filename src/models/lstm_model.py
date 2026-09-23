"""Frozen PyTorch model and training primitives for the Phase 2F LSTM benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only in a broken environment
    raise RuntimeError(
        "Phase 2F requires the preregistered PyTorch dependency (torch>=2,<3)."
    ) from exc


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_SEARCH_PATH = REPOSITORY_ROOT / "configs" / "model_search.yaml"

TARGET_ASSETS: tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "NVDA")
MARKET_REFERENCE = "SPY"
FROZEN_FEATURE_COLUMNS: tuple[str, ...] = (
    "asset_log_return_1d",
    "asset_log_return_lag_1",
    "asset_return_mean_5",
    "asset_return_std_5",
    "asset_volume_mean_5",
    "asset_volume_std_5",
    "asset_return_mean_21",
    "asset_return_std_21",
    "asset_volume_mean_21",
    "asset_volume_std_21",
    "spy_log_return_1d",
    "spy_log_return_lag_1",
    "spy_return_mean_5",
    "spy_volatility_5",
    "spy_return_mean_21",
    "spy_volatility_21",
    "spy_trend_63",
)
CANONICAL_PREDICTION_FIELDS: tuple[str, ...] = (
    "run_id",
    "spec_version",
    "data_version",
    "model",
    "model_config_id",
    "seed",
    "fold",
    "partition",
    "asset",
    "origin_date",
    "target_date",
    "actual_log_return",
    "predicted_log_return",
    "actual_direction",
    "predicted_direction",
    "trend_regime",
    "volatility_regime",
    "transition_regime",
)

HIDDEN_SIZE = 32
NUM_LAYERS = 1
POST_LSTM_DROPOUT = 0.20
OUTPUT_SIZE = 1
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 8
MIN_DELTA = 0.00001
GRADIENT_CLIP_NORM = 1.0
LEARNING_RATE = 0.001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
ADAM_AMSGRAD = False
FROZEN_SEEDS: tuple[int, ...] = (1729, 2718, 31415)
REGIME_PLACEHOLDER = "not_labeled_pre_regime_analysis"


class LSTMContractError(ValueError):
    """Raised when the machine-readable preregistration has drifted."""


class LSTMTrainingError(RuntimeError):
    """Raised when a non-finite or otherwise invalid training result is observed."""


@dataclass(frozen=True)
class LSTMCandidate:
    config_id: str
    context_length: int


FROZEN_CANDIDATES: tuple[LSTMCandidate, ...] = (
    LSTMCandidate("LSTM_21", 21),
    LSTMCandidate("LSTM_63", 63),
)


@dataclass(frozen=True)
class LSTMContract:
    spec_version: str
    raw_sha256: str
    processed_sha256: str
    feature_columns: tuple[str, ...]
    assets: tuple[str, ...]
    candidates: tuple[LSTMCandidate, ...]
    seeds: tuple[int, ...]
    prediction_fields: tuple[str, ...]
    regime_placeholder: str


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LSTMContractError(f"{path} must be a mapping.")
    return value


def _at(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        current = _mapping(current, path)[part]
    return current


def _expect(payload: Mapping[str, Any], path: str, expected: Any) -> None:
    observed = _at(payload, path)
    if observed != expected:
        raise LSTMContractError(
            f"Frozen model-search field {path} drifted: expected={expected!r}, "
            f"observed={observed!r}."
        )


def load_lstm_contract(path: str | Path | None = None) -> LSTMContract:
    """Load and validate the exact frozen LSTM contract."""

    config_path = Path(path) if path is not None else DEFAULT_MODEL_SEARCH_PATH
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise LSTMContractError(f"Unable to read {config_path}: {exc}") from exc
    payload = _mapping(payload, "root")

    checks: tuple[tuple[str, Any], ...] = (
        ("schema_version", 1),
        ("status", "FROZEN_BEFORE_LEARNED_MODEL_TRAINING"),
        ("experiment_specification.version", "1.1"),
        ("features.count", 17),
        ("features.columns", list(FROZEN_FEATURE_COLUMNS)),
        ("assets.target_assets", list(TARGET_ASSETS)),
        ("assets.market_reference", MARKET_REFERENCE),
        ("assets.market_reference_is_prediction_target", False),
        ("candidate_selection.primary_statistic", "equal_weight_macro_validation_mae"),
        ("candidate_selection.tie_breaking.tolerance", MIN_DELTA),
        ("candidate_selection.tie_breaking.family_simplicity_order.lstm", ["LSTM_21", "LSTM_63"]),
        ("lstm.framework", "PyTorch"),
        ("lstm.architecture.input_feature_count", 17),
        ("lstm.architecture.batch_first", True),
        ("lstm.architecture.hidden_size", HIDDEN_SIZE),
        ("lstm.architecture.num_layers", NUM_LAYERS),
        ("lstm.architecture.bidirectional", False),
        ("lstm.architecture.recurrent_dropout", 0.0),
        ("lstm.architecture.post_lstm_dropout", POST_LSTM_DROPOUT),
        ("lstm.architecture.output_size", OUTPUT_SIZE),
        ("lstm.architecture.attention", False),
        ("lstm.architecture.stacked_lstm", False),
        ("lstm.architecture.tensor_and_model_dtype", "float32"),
        ("lstm.optimization.optimizer", "Adam"),
        ("lstm.optimization.learning_rate", LEARNING_RATE),
        ("lstm.optimization.adam_betas", list(ADAM_BETAS)),
        ("lstm.optimization.adam_eps", ADAM_EPS),
        ("lstm.optimization.adam_weight_decay", ADAM_WEIGHT_DECAY),
        ("lstm.optimization.adam_amsgrad", ADAM_AMSGRAD),
        ("lstm.optimization.loss", "L1Loss"),
        ("lstm.optimization.batch_size", BATCH_SIZE),
        ("lstm.optimization.max_epochs", MAX_EPOCHS),
        ("lstm.optimization.gradient_clip_norm", GRADIENT_CLIP_NORM),
        ("lstm.optimization.shuffle_training_sequences", False),
        ("lstm.optimization.drop_last_training_batch", False),
        ("lstm.optimization.data_loader_num_workers", 0),
        ("lstm.optimization.target_scaling", "none"),
        ("lstm.early_stopping.patience_epochs", PATIENCE),
        ("lstm.early_stopping.min_delta", MIN_DELTA),
        ("lstm.early_stopping.restore_best_validation_epoch", True),
        ("lstm.early_stopping.best_epoch_numbering", "one_based"),
        ("lstm.candidate_count", 2),
        ("lstm.seeds", list(FROZEN_SEEDS)),
        ("lstm.seed_count", 3),
        ("lstm.preprocessing.method", "per_feature_standardization"),
        ("lstm.preprocessing.statistics.standard_deviation_ddof", 0),
        ("lstm.preprocessing.target_standardization", False),
        ("lstm.preprocessing.zero_variance_feature_policy.scale_value", 1.0),
        ("lstm.sequences.asset_local", True),
        ("lstm.sequences.context_includes_origin_date", True),
        ("lstm.sequences.target_values_as_inputs", "prohibited"),
        ("prediction_artifact_contract.canonical_fields", list(CANONICAL_PREDICTION_FIELDS)),
        ("prediction_artifact_contract.regime_placeholders.value", REGIME_PLACEHOLDER),
        ("walk_forward.final_test.status", "LOCKED_UNTOUCHED"),
        ("walk_forward.final_test.generate_2025_learned_predictions_during_model_development", False),
        ("patchtst.status", "CLOSED"),
        ("patchtst.implemented", False),
    )
    for key, expected in checks:
        _expect(payload, key, expected)

    raw_candidates = _at(payload, "lstm.candidates")
    candidates = tuple(
        LSTMCandidate(str(item["config_id"]), int(item["context_length_sessions"]))
        for item in raw_candidates
    )
    if candidates != FROZEN_CANDIDATES:
        raise LSTMContractError(f"LSTM candidates drifted: {candidates!r}.")

    return LSTMContract(
        spec_version=str(_at(payload, "experiment_specification.version")),
        raw_sha256=str(_at(payload, "data_contract.raw_dataset_sha256")),
        processed_sha256=str(_at(payload, "data_contract.processed_dataset_sha256")),
        feature_columns=tuple(_at(payload, "features.columns")),
        assets=tuple(_at(payload, "assets.target_assets")),
        candidates=candidates,
        seeds=tuple(int(seed) for seed in _at(payload, "lstm.seeds")),
        prediction_fields=tuple(_at(payload, "prediction_artifact_contract.canonical_fields")),
        regime_placeholder=str(_at(payload, "prediction_artifact_contract.regime_placeholders.value")),
    )


def configure_cpu_runtime() -> dict[str, Any]:
    """Apply the frozen single-thread CPU and strict deterministic policy."""

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise LSTMTrainingError(
                "PyTorch inter-op threads could not be frozen at one before parallel work."
            ) from exc
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return {
        "device": "cpu",
        "intra_op_threads": int(torch.get_num_threads()),
        "inter_op_threads": int(torch.get_num_interop_threads()),
        "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch CPU before each fresh model."""

    if seed not in FROZEN_SEEDS:
        raise ValueError(f"Seed must be one of {FROZEN_SEEDS}, got {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMRegressor(nn.Module):
    """The exact preregistered one-layer LSTM regression architecture."""

    def __init__(self, *, context_length: int) -> None:
        super().__init__()
        if context_length not in {candidate.context_length for candidate in FROZEN_CANDIDATES}:
            raise ValueError("context_length must be 21 or 63.")
        self.context_length = int(context_length)
        self.lstm = nn.LSTM(
            input_size=len(FROZEN_FEATURE_COLUMNS),
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.post_lstm_dropout = nn.Dropout(p=POST_LSTM_DROPOUT)
        self.output = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.to(dtype=torch.float32, device=torch.device("cpu"))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("LSTM input must have shape (batch, context, features).")
        if inputs.shape[1:] != (self.context_length, len(FROZEN_FEATURE_COLUMNS)):
            raise ValueError(
                "LSTM input shape does not match the frozen context and feature count."
            )
        outputs, _ = self.lstm(inputs)
        representation = outputs[:, -1, :]
        return self.output(self.post_lstm_dropout(representation)).squeeze(-1)


def make_loader(
    features: np.ndarray,
    targets: np.ndarray | None = None,
    *,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build a chronological, non-dropping, single-worker data loader."""

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 3 or x.shape[2] != len(FROZEN_FEATURE_COLUMNS):
        raise ValueError("features must have shape (n, context, 17).")
    tensors: tuple[torch.Tensor, ...]
    if targets is None:
        tensors = (torch.from_numpy(x),)
    else:
        y = np.asarray(targets, dtype=np.float32)
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("targets must be a one-dimensional array aligned to features.")
        tensors = (torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def build_optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
        weight_decay=ADAM_WEIGHT_DECAY,
        amsgrad=ADAM_AMSGRAD,
    )


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_one_epoch(
    model: LSTMRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train exactly one chronological epoch and return row-weighted MAE loss."""

    model.train()
    loss_function = nn.L1Loss(reduction="mean")
    absolute_error_sum = 0.0
    n_rows = 0
    for batch in loader:
        if len(batch) != 2:
            raise ValueError("Training loader must contain features and targets.")
        inputs, targets = (tensor.to(device="cpu", dtype=torch.float32) for tensor in batch)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        if not torch.isfinite(predictions).all():
            raise LSTMTrainingError("Non-finite training prediction.")
        loss = loss_function(predictions, targets)
        if not torch.isfinite(loss):
            raise LSTMTrainingError("Non-finite training loss.")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise LSTMTrainingError("Non-finite gradient.")
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        if not torch.isfinite(total_norm):
            raise LSTMTrainingError("Non-finite gradient norm.")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise LSTMTrainingError("Non-finite model parameter after optimizer step.")
        absolute_error_sum += float(torch.abs(predictions.detach() - targets).sum().item())
        n_rows += int(targets.numel())
    if n_rows == 0:
        raise ValueError("Training loader is empty.")
    return absolute_error_sum / n_rows


def predict(model: LSTMRegressor, features: np.ndarray) -> np.ndarray:
    """Generate deterministic CPU predictions with dropout disabled."""

    loader = make_loader(features)
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for (inputs,) in loader:
            values = model(inputs.to(device="cpu", dtype=torch.float32))
            if not torch.isfinite(values).all():
                raise LSTMTrainingError("Non-finite evaluation prediction.")
            parts.append(values.detach().cpu().numpy().astype(np.float64, copy=False))
    if not parts:
        raise ValueError("Prediction input is empty.")
    return np.concatenate(parts)


def train_fixed_epochs(
    model: LSTMRegressor,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
) -> tuple[float, ...]:
    """Train a fresh refit model for exactly the recorded one-based epoch count."""

    if not 1 <= int(epochs) <= MAX_EPOCHS:
        raise ValueError(f"epochs must be in [1, {MAX_EPOCHS}].")
    loader = make_loader(features, targets)
    optimizer = build_optimizer(model)
    return tuple(train_one_epoch(model, loader, optimizer) for _ in range(int(epochs)))
