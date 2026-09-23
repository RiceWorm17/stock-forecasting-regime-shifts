"""Frozen PatchTST-style model and training primitives for Phase 2H-B.

This module deliberately exposes only the bounded-pilot model.  It has no
development-test, full-benchmark, final-test, or regime-analysis entry point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import yaml

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - broken environment only
    raise RuntimeError(
        "Phase 2H-B requires the already-existing frozen PyTorch CPU environment."
    ) from exc


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATCHTST_SEARCH_PATH = REPOSITORY_ROOT / "configs" / "patchtst_search.yaml"

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

PATCH_LENGTH = 16
PATCH_STRIDE = 8
D_MODEL = 32
N_HEADS = 4
NUM_ENCODER_LAYERS = 2
FEEDFORWARD_DIM = 64
TRANSFORMER_DROPOUT = 0.10
HEAD_DROPOUT = 0.10
LAYER_NORM_EPS = 1e-5
OUTPUT_SIZE = 1

BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 6
MIN_DELTA = 1e-5
GRADIENT_CLIP_NORM = 1.0
LEARNING_RATE = 0.001
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.0001
ADAMW_AMSGRAD = False

PILOT_SEEDS: tuple[int, ...] = (1729, 2718)
FUTURE_FULL_SEEDS: tuple[int, ...] = (1729, 2718, 31415)


class PatchTSTContractError(ValueError):
    """Raised when the machine-readable frozen contract has drifted."""


class PatchTSTTrainingError(RuntimeError):
    """Raised for a non-finite or nondeterministic training condition."""


@dataclass(frozen=True)
class PatchTSTCandidate:
    config_id: str
    context_length: int
    patch_start_indices: tuple[int, ...]
    patch_count: int
    head_input_dimension: int


FROZEN_CANDIDATES: tuple[PatchTSTCandidate, ...] = (
    PatchTSTCandidate(
        config_id="PATCHTST_63",
        context_length=63,
        patch_start_indices=(7, 15, 23, 31, 39, 47),
        patch_count=6,
        head_input_dimension=3264,
    ),
    PatchTSTCandidate(
        config_id="PATCHTST_126",
        context_length=126,
        patch_start_indices=(6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110),
        patch_count=14,
        head_input_dimension=7616,
    ),
)


@dataclass(frozen=True)
class PatchTSTContract:
    raw_sha256: str
    processed_sha256: str
    feature_columns: tuple[str, ...]
    assets: tuple[str, ...]
    candidates: tuple[PatchTSTCandidate, ...]
    pilot_seeds: tuple[int, ...]
    future_full_seeds: tuple[int, ...]
    frozen_trees: Mapping[str, Mapping[str, Any]]
    environment_versions: Mapping[str, Any]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PatchTSTContractError(f"{path} must be a mapping.")
    return value


def _at(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    try:
        for part in path.split("."):
            current = _mapping(current, path)[part]
    except KeyError as exc:
        raise PatchTSTContractError(f"Frozen PatchTST field is missing: {path}.") from exc
    return current


def _expect(payload: Mapping[str, Any], path: str, expected: Any) -> None:
    observed = _at(payload, path)
    if observed != expected:
        raise PatchTSTContractError(
            f"Frozen PatchTST field {path} drifted: expected={expected!r}, observed={observed!r}."
        )


def right_aligned_patch_starts(context_length: int) -> tuple[int, ...]:
    """Return the exact chronological no-padding right-aligned patch grid."""

    frozen_contexts = {candidate.context_length for candidate in FROZEN_CANDIDATES}
    if int(context_length) not in frozen_contexts:
        raise ValueError(f"context_length must be one of {sorted(frozen_contexts)}.")
    starts: list[int] = []
    current = int(context_length) - PATCH_LENGTH
    while current >= 0:
        starts.append(current)
        current -= PATCH_STRIDE
    starts.reverse()
    result = tuple(starts)
    candidate = next(item for item in FROZEN_CANDIDATES if item.context_length == context_length)
    if result != candidate.patch_start_indices:
        raise AssertionError("Computed right-aligned patch grid differs from the frozen candidate.")
    return result


def load_patchtst_contract(path: str | Path | None = None) -> PatchTSTContract:
    """Load and exhaustively validate the Phase 2H-A frozen YAML contract."""

    config_path = Path(path) if path is not None else DEFAULT_PATCHTST_SEARCH_PATH
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PatchTSTContractError(f"Unable to read {config_path}: {exc}") from exc
    payload = _mapping(payload, "root")

    checks: tuple[tuple[str, Any], ...] = (
        ("schema_version", 1),
        ("artifact_type", "patchtst_preregistration"),
        ("status", "FROZEN_BEFORE_FIRST_PATCHTST_FIT"),
        ("authorization.phase", "PHASE_2H_A"),
        ("authorization.purpose", "preregistration_only"),
        ("authorization.authorizes_implementation_now", False),
        ("authorization.authorizes_training_now", False),
        ("authorization.authorizes_pilot_now", False),
        ("authorization.authorizes_full_d1_d5_benchmark", False),
        ("prediction_task.target_column", "target_log_return"),
        ("prediction_task.forecast_horizon_observed_sessions", 1),
        ("prediction_task.target_scaling", "none"),
        ("prediction_task.secondary_direction.rule", "1 if predicted_log_return > 0 else 0"),
        ("prediction_task.secondary_direction.separate_classifier", "prohibited"),
        ("assets.target_assets", list(TARGET_ASSETS)),
        ("assets.market_reference", MARKET_REFERENCE),
        ("assets.market_reference_is_prediction_target", False),
        ("features.count", 17),
        ("features.order", list(FROZEN_FEATURE_COLUMNS)),
        ("features.additional_features_allowed", False),
        ("features.regime_labels_are_features", False),
        ("model.family", "patchtst"),
        ("model.implementation", "small_custom_pytorch"),
        ("model.defining_properties.channel_independent_patch_encoding", True),
        ("patching.alignment", "right"),
        ("patching.padding", "none"),
        ("patching.patch_length", PATCH_LENGTH),
        ("patching.stride", PATCH_STRIDE),
        ("patching.final_patch_must_include_origin_timestep", True),
        ("architecture.input_channels", 17),
        ("architecture.patch_projection.type", "linear"),
        ("architecture.patch_projection.in_features", PATCH_LENGTH),
        ("architecture.patch_projection.out_features", D_MODEL),
        ("architecture.patch_projection.bias", True),
        ("architecture.patch_projection.shared_across_channels", True),
        ("architecture.positional_embedding.type", "learned"),
        ("architecture.positional_embedding.initialization", "zeros"),
        ("architecture.positional_embedding.shared_across_channels", True),
        ("architecture.transformer_encoder.d_model", D_MODEL),
        ("architecture.transformer_encoder.n_heads", N_HEADS),
        ("architecture.transformer_encoder.layers", NUM_ENCODER_LAYERS),
        ("architecture.transformer_encoder.feedforward_dimension", FEEDFORWARD_DIM),
        ("architecture.transformer_encoder.dropout", TRANSFORMER_DROPOUT),
        ("architecture.transformer_encoder.activation", "GELU"),
        ("architecture.transformer_encoder.batch_first", True),
        ("architecture.transformer_encoder.norm_first", True),
        ("architecture.transformer_encoder.layer_norm_eps", LAYER_NORM_EPS),
        ("architecture.transformer_encoder.final_encoder_layer_norm", True),
        ("architecture.transformer_encoder.weights_shared_across_channels", True),
        ("architecture.head.pooling", "none"),
        ("architecture.head.attention_pooling", False),
        ("architecture.head.dropout", HEAD_DROPOUT),
        ("architecture.head.output_size", OUTPUT_SIZE),
        ("architecture.dtype", "float32"),
        ("architecture.revin", False),
        ("architecture.alternate_head_search", False),
        ("architecture.architecture_search", False),
        ("candidates.count", 2),
        ("candidates.additional_contexts_allowed", False),
        ("candidates.searched_fields", []),
        ("preprocessing.target_standardization", False),
        ("preprocessing.candidate_stage.unit", "fold_x_asset"),
        ("preprocessing.candidate_stage.fit_rows", "training_feature_rows_only"),
        ("preprocessing.candidate_stage.scaler_shared_across_context_candidates", True),
        ("preprocessing.candidate_stage.scaler_shared_across_seeds", True),
        ("preprocessing.statistics.ddof", 0),
        ("preprocessing.zero_variance_policy.effective_scale", 1.0),
        ("sequences.asset_local", True),
        ("sequences.context_ends_at_origin_date", True),
        ("sequences.target_as_input", "prohibited"),
        ("sequences.padding", "none"),
        ("training.loss.implementation", "torch.nn.L1Loss"),
        ("training.loss.reduction", "mean"),
        ("training.optimizer.implementation", "torch.optim.AdamW"),
        ("training.optimizer.learning_rate", LEARNING_RATE),
        ("training.optimizer.betas", list(ADAMW_BETAS)),
        ("training.optimizer.eps", ADAMW_EPS),
        ("training.optimizer.weight_decay", ADAMW_WEIGHT_DECAY),
        ("training.optimizer.amsgrad", ADAMW_AMSGRAD),
        ("training.batch_size", BATCH_SIZE),
        ("training.max_epochs", MAX_EPOCHS),
        ("training.shuffle", False),
        ("training.drop_last", False),
        ("training.num_workers", 0),
        ("training.gradient_clip_norm", GRADIENT_CLIP_NORM),
        ("training.scheduler", "none"),
        ("training.warmup_scheduler", "none"),
        ("training.early_stopping.min_delta", MIN_DELTA),
        ("training.early_stopping.patience_completed_epochs", PATIENCE),
        ("training.early_stopping.restore_best_state", True),
        ("determinism.reference_device", "cpu"),
        ("determinism.pytorch_intra_op_threads", 1),
        ("determinism.pytorch_inter_op_threads", 1),
        ("pilot.fold", "D1"),
        ("pilot.seeds", list(PILOT_SEEDS)),
        ("pilot.grid.candidate_fits", 16),
        ("pilot.grid.refits", 0),
        ("pilot.grid.test_predictions", 0),
        ("pilot.deterministic_replay.configuration", "PATCHTST_63"),
        ("pilot.deterministic_replay.asset", "AAPL"),
        ("pilot.deterministic_replay.seed", 1729),
        ("pilot.deterministic_replay.total_training_executions_including_replay", 17),
        ("pilot_acceptance_gate.predictive_performance_thresholds", []),
        ("pilot_acceptance_gate.predictive_performance_used_for_acceptance", False),
        ("resource_budget.pilot_candidate_fit_budget", 16),
        ("resource_budget.pilot_total_training_executions_including_replay", 17),
        ("resource_budget.pilot_wall_time_cap_seconds", 1800),
        ("resource_budget.full_runtime_estimate_formula", "(pilot_total_seconds / 16) * 180 * 1.25"),
        ("resource_budget.full_runtime_cap_seconds", 21600),
        ("dependency_policy.phase_2h_a_installs_nothing", True),
        ("scope_guards.f1_2025_performance_access", "prohibited"),
        ("scope_guards.regime_analysis", "prohibited"),
        ("scope_guards.phase_2h_a_model_implementation", "prohibited"),
        ("scope_guards.phase_2h_a_training", "prohibited"),
    )
    for key, expected in checks:
        _expect(payload, key, expected)

    raw_candidates = _at(payload, "candidates.values")
    candidates = tuple(
        PatchTSTCandidate(
            config_id=str(item["config_id"]),
            context_length=int(item["context_length"]),
            patch_start_indices=tuple(int(value) for value in item["chronological_patch_start_indices"]),
            patch_count=int(item["patch_count"]),
            head_input_dimension=int(item["head_input_dimension"]),
        )
        for item in raw_candidates
    )
    if candidates != FROZEN_CANDIDATES:
        raise PatchTSTContractError(f"PatchTST candidates drifted: {candidates!r}.")
    for candidate in candidates:
        if right_aligned_patch_starts(candidate.context_length) != candidate.patch_start_indices:
            raise PatchTSTContractError(f"Patch grid drifted for {candidate.config_id}.")
        if candidate.head_input_dimension != 17 * candidate.patch_count * D_MODEL:
            raise PatchTSTContractError(f"Head width drifted for {candidate.config_id}.")

    gates = _mapping(_at(payload, "pilot_acceptance_gate.criteria"), "pilot gate")
    if tuple(gates) != ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        raise PatchTSTContractError("Pilot gate must contain exactly P1 through P7 in order.")

    return PatchTSTContract(
        raw_sha256=str(_at(payload, "provenance.data_versions.raw_dataset_sha256")),
        processed_sha256=str(_at(payload, "provenance.data_versions.processed_dataset_sha256")),
        feature_columns=tuple(_at(payload, "features.order")),
        assets=tuple(_at(payload, "assets.target_assets")),
        candidates=candidates,
        pilot_seeds=tuple(int(value) for value in _at(payload, "pilot.seeds")),
        future_full_seeds=tuple(int(value) for value in _at(payload, "future_full_benchmark.seeds")),
        frozen_trees=_mapping(_at(payload, "provenance.frozen_artifact_trees"), "frozen trees"),
        environment_versions=_mapping(
            _at(payload, "determinism.exact_reference_environment"), "reference environment"
        ),
    )


def configure_cpu_runtime() -> dict[str, Any]:
    """Freeze the preregistered single-thread deterministic CPU policy."""

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise PatchTSTTrainingError(
                "PyTorch inter-op threads could not be frozen before parallel work."
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

    if int(seed) not in FUTURE_FULL_SEEDS:
        raise ValueError(f"Seed must be one of {FUTURE_FULL_SEEDS}, got {seed}.")
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def extract_right_aligned_patches(inputs: torch.Tensor) -> torch.Tensor:
    """Extract exact patches as ``(batch, channel, patch, patch_value)``."""

    if inputs.ndim != 3:
        raise ValueError("PatchTST input must have shape (batch, context, features).")
    if inputs.shape[2] != len(FROZEN_FEATURE_COLUMNS):
        raise ValueError("PatchTST input must contain exactly 17 ordered features.")
    starts = right_aligned_patch_starts(int(inputs.shape[1]))
    channel_time = inputs.transpose(1, 2)
    patches = torch.stack(
        [channel_time[..., start : start + PATCH_LENGTH] for start in starts], dim=2
    )
    expected = (inputs.shape[0], len(FROZEN_FEATURE_COLUMNS), len(starts), PATCH_LENGTH)
    if tuple(patches.shape) != expected:
        raise AssertionError(f"Patch extraction produced {tuple(patches.shape)}, expected {expected}.")
    return patches


class PatchTSTRegressor(nn.Module):
    """Exact small channel-independent patch Transformer frozen for the pilot."""

    def __init__(self, *, context_length: int) -> None:
        super().__init__()
        candidate = next(
            (item for item in FROZEN_CANDIDATES if item.context_length == int(context_length)),
            None,
        )
        if candidate is None:
            raise ValueError("context_length must be 63 or 126.")
        self.context_length = candidate.context_length
        self.patch_start_indices = candidate.patch_start_indices
        self.patch_count = candidate.patch_count
        self.head_input_dimension = candidate.head_input_dimension

        self.patch_projection = nn.Linear(PATCH_LENGTH, D_MODEL, bias=True)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.patch_count, D_MODEL))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=TRANSFORMER_DROPOUT,
            activation="gelu",
            layer_norm_eps=LAYER_NORM_EPS,
            batch_first=True,
            norm_first=True,
            bias=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_ENCODER_LAYERS,
            norm=nn.LayerNorm(D_MODEL, eps=LAYER_NORM_EPS),
            enable_nested_tensor=False,
        )
        self.head_dropout = nn.Dropout(p=HEAD_DROPOUT)
        self.output = nn.Linear(self.head_input_dimension, OUTPUT_SIZE)
        self.to(device=torch.device("cpu"), dtype=torch.float32)

    def encode_channels(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return encoded values shaped ``(batch, channel, patch, d_model)``."""

        if inputs.dtype != torch.float32 or inputs.device.type != "cpu":
            raise ValueError("PatchTST inputs must be CPU float32 tensors.")
        patches = extract_right_aligned_patches(inputs)
        batch, channels, patch_count, _ = patches.shape
        channel_samples = patches.reshape(batch * channels, patch_count, PATCH_LENGTH)
        embedded = self.patch_projection(channel_samples)
        embedded = embedded + self.positional_embedding
        encoded = self.transformer_encoder(embedded)
        return encoded.reshape(batch, channels, patch_count, D_MODEL)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (
            self.context_length,
            len(FROZEN_FEATURE_COLUMNS),
        ):
            raise ValueError("PatchTST input shape differs from the frozen context/features.")
        encoded = self.encode_channels(inputs)
        flattened = encoded.reshape(inputs.shape[0], self.head_input_dimension)
        return self.output(self.head_dropout(flattened)).squeeze(-1)


def make_loader(
    features: np.ndarray,
    targets: np.ndarray | None = None,
    *,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Build a chronological, non-dropping, zero-worker CPU data loader."""

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] not in {63, 126} or x.shape[2] != 17:
        raise ValueError("features must have shape (n, 63|126, 17).")
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
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
        amsgrad=ADAMW_AMSGRAD,
    )


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_one_epoch(
    model: PatchTSTRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train one chronological epoch and return row-weighted mean absolute error."""

    model.train()
    loss_function = nn.L1Loss(reduction="mean")
    absolute_error_sum = 0.0
    n_rows = 0
    for batch in loader:
        if len(batch) != 2:
            raise ValueError("Training loader must contain features and targets.")
        inputs, targets = (value.to(device="cpu", dtype=torch.float32) for value in batch)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        if not torch.isfinite(predictions).all():
            raise PatchTSTTrainingError("Non-finite training prediction.")
        loss = loss_function(predictions, targets)
        if not torch.isfinite(loss):
            raise PatchTSTTrainingError("Non-finite training loss.")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise PatchTSTTrainingError("Non-finite gradient.")
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        if not torch.isfinite(total_norm):
            raise PatchTSTTrainingError("Non-finite gradient norm.")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise PatchTSTTrainingError("Non-finite parameter after optimizer step.")
        absolute_error_sum += float(torch.abs(predictions.detach() - targets).sum().item())
        n_rows += int(targets.numel())
    if n_rows == 0:
        raise ValueError("Training loader is empty.")
    return absolute_error_sum / n_rows


def predict(model: PatchTSTRegressor, features: np.ndarray) -> np.ndarray:
    """Generate deterministic CPU predictions with dropout disabled."""

    loader = make_loader(features)
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for (inputs,) in loader:
            values = model(inputs.to(device="cpu", dtype=torch.float32))
            if not torch.isfinite(values).all():
                raise PatchTSTTrainingError("Non-finite evaluation prediction.")
            pieces.append(values.detach().cpu().numpy().astype(np.float64, copy=False))
    if not pieces:
        raise ValueError("Prediction input is empty.")
    return np.concatenate(pieces)
