"""Leakage-aware sequence, fitting, selection, and scoring logic for Phase 2F."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from src.evaluation.walk_forward import DEVELOPMENT_FOLD_IDS, FoldPartitions, FoldRegistry, split_samples_for_fold
from src.models.lstm_model import (
    BATCH_SIZE,
    CANONICAL_PREDICTION_FIELDS,
    FROZEN_CANDIDATES,
    FROZEN_FEATURE_COLUMNS,
    FROZEN_SEEDS,
    GRADIENT_CLIP_NORM,
    HIDDEN_SIZE,
    LSTMContract,
    LSTMCandidate,
    LSTMRegressor,
    MAX_EPOCHS,
    MIN_DELTA,
    NUM_LAYERS,
    OUTPUT_SIZE,
    PATIENCE,
    POST_LSTM_DROPOUT,
    REGIME_PLACEHOLDER,
    TARGET_ASSETS,
    build_optimizer,
    clone_state_dict,
    make_loader,
    predict,
    seed_everything,
    train_fixed_epochs,
    train_one_epoch,
)


SAMPLE_KEYS: tuple[str, ...] = ("asset", "origin_date", "target_date")
MATCH_KEYS: tuple[str, ...] = ("fold", "asset", "origin_date", "target_date")
CANDIDATE_PREDICTION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "model_config_id",
    "context_length",
    "asset",
    "seed",
    "origin_date",
    "target_date",
    "actual_log_return",
    "predicted_log_return",
    "actual_direction",
    "predicted_direction",
)
CANDIDATE_AUDIT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "model_config_id",
    "context_length",
    "asset",
    "seed",
    "training_scaler_identifier",
    "n_train_endpoints",
    "n_validation_endpoints",
    "best_epoch",
    "epochs_run",
    "stopping_reason",
    "validation_mae",
    "validation_rmse",
    "validation_directional_accuracy",
    "actual_positive_rate",
    "fit_runtime_seconds",
    "completion_status",
    "retry_status",
    "warning_status",
    "warning_messages",
    "error_status",
    "error_type",
    "error_message",
    "candidate_checkpoint_identifier",
)
SELECTION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "model_config_id",
    "context_length",
    "aapl_seed_mean_mae",
    "msft_seed_mean_mae",
    "googl_seed_mean_mae",
    "nvda_seed_mean_mae",
    "aapl_seed_mean_rmse",
    "msft_seed_mean_rmse",
    "googl_seed_mean_rmse",
    "nvda_seed_mean_rmse",
    "macro_validation_mae",
    "macro_validation_rmse",
    "primary_rank",
    "within_practical_tie",
    "selected",
    "selection_reason",
)
BEST_EPOCH_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "winning_model_config_id",
    "context_length",
    "asset",
    "seed",
    "best_epoch",
    "candidate_validation_mae",
    "candidate_checkpoint_identifier",
)
REFIT_AUDIT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "model_config_id",
    "context_length",
    "asset",
    "seed",
    "refit_scaler_identifier",
    "n_refit_endpoints",
    "n_test_endpoints",
    "epochs_requested",
    "epochs_completed",
    "early_stopping_used",
    "fit_runtime_seconds",
    "completion_status",
    "warning_status",
    "warning_messages",
    "error_status",
    "error_type",
    "error_message",
    "refit_model_identifier",
)
METRIC_NAMES: tuple[str, ...] = (
    "mae",
    "rmse",
    "directional_accuracy",
    "actual_positive_rate",
    "baseline_mae",
    "mae_skill",
    "baseline_directional_accuracy",
    "da_difference",
)
PER_SEED_METRIC_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "asset",
    "seed",
    "model_config_id",
    "n",
    *METRIC_NAMES,
)
ASSET_SEED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "asset",
    "model_config_id",
    "n_seeds",
    "n_per_seed",
    *(f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std", "min", "max")),
)
MACRO_BY_SEED_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "seed",
    "model_config_id",
    "n_assets",
    "n_observations",
    *METRIC_NAMES,
)
MACRO_SEED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "model_config_id",
    "n_seeds",
    "n_assets",
    "n_observations_per_seed",
    *(f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std", "min", "max")),
)
DEVELOPMENT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "scope",
    "asset",
    "metric",
    "n_folds",
    "median",
    "q1",
    "q3",
    "iqr",
    "input_statistic",
    "interpretation",
)


class LSTMEvaluationError(RuntimeError):
    """Raised for a Phase 2F coverage, fitting, or artifact-contract failure."""


@dataclass(frozen=True)
class SequenceSet:
    features: np.ndarray
    targets: np.ndarray
    rows: pd.DataFrame
    unavailable_keys: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class ScalerRecord:
    scaler: StandardScaler
    usage: str
    fold: str
    asset: str
    row_count: int
    identifier: str
    payload: bytes


@dataclass(frozen=True)
class CandidateFitResult:
    audit: dict[str, Any]
    predictions: pd.DataFrame
    checkpoint_payload: bytes
    checkpoint_identifier: str
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class RefitResult:
    audit: dict[str, Any]
    predictions: pd.DataFrame
    checkpoint_payload: bytes
    model_identifier: str
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class SelectionResult:
    comparison: pd.DataFrame
    best_epochs: pd.DataFrame
    winners: dict[str, LSTMCandidate]


@dataclass(frozen=True)
class WrittenArtifact:
    path: Path
    sha256: str
    rows: int | None = None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")


def write_immutable(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite immutable artifact {destination}.")
        return
    with destination.open("xb") as handle:
        handle.write(payload)


def write_content_addressed(
    directory: str | Path,
    stem: str,
    suffix: str,
    payload: bytes,
    *,
    rows: int | None = None,
) -> WrittenArtifact:
    digest = sha256_bytes(payload)
    path = Path(directory) / f"{stem}_{digest}{suffix}"
    write_immutable(path, payload)
    return WrittenArtifact(path=path, sha256=digest, rows=rows)


def _boolean(values: pd.Series, name: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="raise")
    if not numeric.isin([0, 1]).all():
        raise LSTMEvaluationError(f"{name} must contain only boolean/0/1 values.")
    return numeric.astype(bool)


def validate_development_samples(samples: pd.DataFrame, *, contract: LSTMContract) -> pd.DataFrame:
    """Validate development rows while retaining early ineligible history for windows."""

    required = set(SAMPLE_KEYS) | {
        "target_log_return",
        "target_direction",
        "core_evaluation_eligible",
        *contract.feature_columns,
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise LSTMEvaluationError(f"Processed samples are missing columns: {missing}.")
    if contract.feature_columns != FROZEN_FEATURE_COLUMNS or contract.assets != TARGET_ASSETS:
        raise LSTMEvaluationError("Loaded contract differs from frozen feature/asset order.")
    frame = samples.copy()
    frame["asset"] = frame["asset"].astype(str)
    if not set(frame["asset"]).issubset(set(TARGET_ASSETS)):
        raise LSTMEvaluationError("Processed samples contain a non-target asset.")
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    if frame[["origin_date", "target_date"]].isna().any().any():
        raise LSTMEvaluationError("Processed samples contain missing dates.")
    if (frame["target_date"] <= frame["origin_date"]).any():
        raise LSTMEvaluationError("Every target date must follow its origin date.")
    if (frame["target_date"].dt.year >= 2025).any():
        raise LSTMEvaluationError("Phase 2F accepts development rows through 2024 only.")
    if frame.duplicated(list(SAMPLE_KEYS), keep=False).any():
        raise LSTMEvaluationError("Processed samples contain duplicate sample identities.")
    frame["core_evaluation_eligible"] = _boolean(
        frame["core_evaluation_eligible"], "core_evaluation_eligible"
    )
    target = pd.to_numeric(frame["target_log_return"], errors="coerce")
    direction = pd.to_numeric(frame["target_direction"], errors="coerce")
    eligible = frame["core_evaluation_eligible"]
    if not np.isfinite(target.loc[eligible]).all() or not direction.loc[eligible].isin([0, 1]).all():
        raise LSTMEvaluationError("Eligible rows contain invalid targets.")
    finite_features = np.isfinite(
        frame.loc[eligible, contract.feature_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    if not finite_features.all():
        raise LSTMEvaluationError("Eligible rows contain non-finite feature values.")
    frame["target_log_return"] = target
    frame["target_direction"] = direction.astype("Int64")
    return frame.sort_values(["asset", "origin_date"], kind="mergesort", ignore_index=True)


def prepare_development_fold(
    samples: pd.DataFrame,
    fold: str,
    *,
    registry: FoldRegistry,
) -> FoldPartitions:
    if fold not in DEVELOPMENT_FOLD_IDS:
        raise ValueError(f"Phase 2F fold must be one of {DEVELOPMENT_FOLD_IDS}.")
    split = split_samples_for_fold(samples, fold, registry=registry, include_test=True)
    return FoldPartitions(
        fold=split.fold,
        train=split.train.loc[split.train["core_evaluation_eligible"]].copy(),
        validation=split.validation.loc[split.validation["core_evaluation_eligible"]].copy(),
        test=split.test.loc[split.test["core_evaluation_eligible"]].copy(),
    )


def _feature_row_hash(rows: pd.DataFrame, feature_columns: Sequence[str]) -> str:
    columns = ["asset", "origin_date", "target_date", *feature_columns]
    canonical = rows.loc[:, columns].sort_values(
        ["asset", "origin_date", "target_date"], kind="mergesort", ignore_index=True
    )
    return sha256_bytes(canonical_csv_bytes(canonical))


def fit_scaler_record(
    rows: pd.DataFrame,
    *,
    fold: str,
    asset: str,
    usage: str,
    feature_columns: Sequence[str],
    data_sha256: str,
    model_config_sha256: str,
) -> ScalerRecord:
    """Fit one StandardScaler on unique canonical rows, never flattened windows."""

    if usage not in {"candidate", "refit"}:
        raise ValueError("usage must be candidate or refit.")
    if tuple(feature_columns) != FROZEN_FEATURE_COLUMNS:
        raise ValueError("Scaler feature order differs from the frozen 17 columns.")
    if rows.empty or set(rows["asset"].astype(str)) != {asset}:
        raise LSTMEvaluationError(f"Scaler rows for {fold}/{asset}/{usage} are invalid.")
    if rows.duplicated(list(SAMPLE_KEYS), keep=False).any():
        raise LSTMEvaluationError("Scaler input contains duplicate canonical feature rows.")
    values = rows.loc[:, feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise LSTMEvaluationError("Scaler input contains non-finite values.")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(values)
    expected_mean = values.mean(axis=0)
    expected_var = values.var(axis=0, ddof=0)
    expected_scale = np.sqrt(expected_var)
    expected_scale[expected_var == 0.0] = 1.0
    if not np.allclose(scaler.mean_, expected_mean, rtol=1e-13, atol=1e-15):
        raise LSTMEvaluationError("StandardScaler mean differs from population mean.")
    if not np.allclose(scaler.var_, expected_var, rtol=1e-12, atol=1e-15):
        raise LSTMEvaluationError("StandardScaler variance differs from population variance.")
    if not np.allclose(scaler.scale_, expected_scale, rtol=1e-12, atol=1e-15):
        raise LSTMEvaluationError("StandardScaler effective scale violates the frozen rule.")

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model_family": "lstm",
        "fold": fold,
        "asset": asset,
        "usage": usage,
        "fit_partition": "train" if usage == "candidate" else "train_plus_validation",
        "feature_order": list(feature_columns),
        "row_count": int(len(rows)),
        "origin_date_start": rows["origin_date"].min().date().isoformat(),
        "origin_date_end": rows["origin_date"].max().date().isoformat(),
        "target_date_start": rows["target_date"].min().date().isoformat(),
        "target_date_end": rows["target_date"].max().date().isoformat(),
        "mean": [float(value) for value in scaler.mean_],
        "variance_ddof_0": [float(value) for value in scaler.var_],
        "scale": [float(value) for value in scaler.scale_],
        "zero_variance_features": [
            feature_columns[index] for index, value in enumerate(scaler.var_) if value == 0.0
        ],
        "canonical_feature_rows_sha256": _feature_row_hash(rows, feature_columns),
        "processed_data_sha256": data_sha256,
        "model_search_sha256": model_config_sha256,
    }
    payload = canonical_json_bytes(metadata)
    identifier = sha256_bytes(payload)
    return ScalerRecord(
        scaler=scaler,
        usage=usage,
        fold=fold,
        asset=asset,
        row_count=len(rows),
        identifier=identifier,
        payload=payload,
    )


def build_sequences(
    history: pd.DataFrame,
    endpoints: pd.DataFrame,
    *,
    context_length: int,
    scaler: StandardScaler,
    require_all_endpoints: bool,
    partition_name: str,
) -> SequenceSet:
    """Build asset-local positional windows ending exactly at each endpoint origin."""

    if context_length not in {21, 63}:
        raise ValueError("context_length must be 21 or 63.")
    if endpoints.empty:
        raise LSTMEvaluationError(f"{partition_name} endpoints are empty.")
    assets = set(endpoints["asset"].astype(str))
    if len(assets) != 1:
        raise LSTMEvaluationError("Sequence endpoints must contain exactly one asset.")
    asset = next(iter(assets))
    asset_history = history.loc[history["asset"].astype(str) == asset].copy()
    asset_history = asset_history.sort_values("origin_date", kind="mergesort", ignore_index=True)
    if asset_history["origin_date"].duplicated().any():
        raise LSTMEvaluationError(f"History for {asset} has duplicate origin dates.")
    position = {pd.Timestamp(value): index for index, value in enumerate(asset_history["origin_date"])}
    raw_features = asset_history.loc[:, FROZEN_FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    windows: list[np.ndarray] = []
    targets: list[float] = []
    row_records: list[dict[str, Any]] = []
    unavailable: list[tuple[str, str, str]] = []
    ordered_endpoints = endpoints.sort_values("origin_date", kind="mergesort", ignore_index=True)
    for row in ordered_endpoints.itertuples(index=False):
        origin = pd.Timestamp(row.origin_date)
        target_date = pd.Timestamp(row.target_date)
        key = (asset, origin.date().isoformat(), target_date.date().isoformat())
        end = position.get(origin)
        start = -1 if end is None else end - context_length + 1
        if end is None or start < 0:
            unavailable.append(key)
            continue
        raw_window = raw_features[start : end + 1]
        if raw_window.shape != (context_length, len(FROZEN_FEATURE_COLUMNS)) or not np.isfinite(raw_window).all():
            unavailable.append(key)
            continue
        if pd.Timestamp(asset_history.iloc[end]["origin_date"]) != origin:
            raise AssertionError("Sequence does not end at its endpoint origin date.")
        transformed = scaler.transform(raw_window)
        if not np.isfinite(transformed).all():
            raise LSTMEvaluationError("Scaler produced non-finite sequence values.")
        target = float(row.target_log_return)
        actual_direction = int(row.target_direction)
        if not np.isfinite(target) or actual_direction not in {0, 1}:
            raise LSTMEvaluationError("Sequence endpoint contains an invalid target.")
        windows.append(transformed.astype(np.float32, copy=False))
        targets.append(target)
        row_records.append(
            {
                "asset": asset,
                "origin_date": origin,
                "target_date": target_date,
                "actual_log_return": target,
                "actual_direction": actual_direction,
            }
        )
    if require_all_endpoints and unavailable:
        raise LSTMEvaluationError(
            f"{partition_name} has {len(unavailable)} nonconstructible required keys; "
            f"first={unavailable[0]}."
        )
    if not windows:
        raise LSTMEvaluationError(f"No constructible {partition_name} sequences for {asset}.")
    rows = pd.DataFrame(row_records)
    return SequenceSet(
        features=np.stack(windows).astype(np.float32, copy=False),
        targets=np.asarray(targets, dtype=np.float32),
        rows=rows,
        unavailable_keys=tuple(unavailable),
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise LSTMEvaluationError("Metric arrays must be aligned, finite vectors.")
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean(np.square(y - p)))),
        "directional_accuracy": float(np.mean((p > 0).astype(np.int8) == (y > 0).astype(np.int8))),
        "actual_positive_rate": float(np.mean(y > 0)),
    }


def _checkpoint_bytes(state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "metadata": dict(metadata),
            "state_dict": {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()},
        },
        buffer,
    )
    return buffer.getvalue()


def fit_candidate(
    train: SequenceSet,
    validation: SequenceSet,
    *,
    run_id: str,
    fold: str,
    candidate: LSTMCandidate,
    asset: str,
    seed: int,
    scaler_identifier: str,
    data_sha256: str,
    model_config_sha256: str,
    retry_status: str = "not_retried",
    max_epochs: int = MAX_EPOCHS,
) -> CandidateFitResult:
    """Fit one seed with exact early stopping and restore its best epoch."""

    if max_epochs != MAX_EPOCHS:
        raise ValueError("Real candidate fits must use the frozen 60-epoch maximum.")
    started = perf_counter()
    seed_everything(seed)
    model = LSTMRegressor(context_length=candidate.context_length)
    optimizer = build_optimizer(model)
    loader = make_loader(train.features, train.targets)
    best_mae = np.inf
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    non_improving = 0
    epochs_run = 0
    stopping_reason = "maximum_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        train_one_epoch(model, loader, optimizer)
        validation_predictions = predict(model, validation.features)
        metrics = regression_metrics(validation.targets, validation_predictions)
        epochs_run = epoch
        if metrics["mae"] < best_mae - MIN_DELTA:
            best_mae = metrics["mae"]
            best_epoch = epoch
            best_metrics = metrics
            best_state = clone_state_dict(model)
            non_improving = 0
        else:
            non_improving += 1
        if non_improving >= PATIENCE:
            stopping_reason = "patience_exhausted"
            break
    if best_state is None or best_metrics is None or not 1 <= best_epoch <= MAX_EPOCHS:
        raise LSTMEvaluationError("Candidate fit did not retain a valid best epoch.")
    model.load_state_dict(best_state, strict=True)
    restored_predictions = predict(model, validation.features)
    restored_metrics = regression_metrics(validation.targets, restored_predictions)
    for metric in ("mae", "rmse", "directional_accuracy", "actual_positive_rate"):
        if not np.isclose(restored_metrics[metric], best_metrics[metric], rtol=0.0, atol=1e-15):
            raise LSTMEvaluationError("Restored candidate state does not reproduce best metrics.")

    checkpoint_metadata = {
        "schema_version": 1,
        "stage": "candidate_best_epoch",
        "run_id": run_id,
        "fold": fold,
        "model_config_id": candidate.config_id,
        "context_length": candidate.context_length,
        "asset": asset,
        "seed": seed,
        "best_epoch": best_epoch,
        "feature_order": list(FROZEN_FEATURE_COLUMNS),
        "architecture": {
            "input_features": 17,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "bidirectional": False,
            "internal_dropout": 0.0,
            "post_lstm_dropout": POST_LSTM_DROPOUT,
            "output_size": OUTPUT_SIZE,
            "dtype": "float32",
        },
        "scaler_identifier": scaler_identifier,
        "processed_data_sha256": data_sha256,
        "model_search_sha256": model_config_sha256,
    }
    checkpoint_payload = _checkpoint_bytes(best_state, checkpoint_metadata)
    checkpoint_identifier = sha256_bytes(checkpoint_payload)
    predictions = validation.rows.copy()
    predictions.insert(0, "seed", seed)
    predictions.insert(0, "context_length", candidate.context_length)
    predictions.insert(0, "model_config_id", candidate.config_id)
    predictions.insert(0, "fold", fold)
    predictions.insert(0, "run_id", run_id)
    predictions["predicted_log_return"] = restored_predictions
    predictions["predicted_direction"] = (restored_predictions > 0).astype(np.int8)
    predictions = predictions.loc[:, CANDIDATE_PREDICTION_COLUMNS]
    runtime = perf_counter() - started
    audit = {
        "run_id": run_id,
        "fold": fold,
        "model_config_id": candidate.config_id,
        "context_length": candidate.context_length,
        "asset": asset,
        "seed": seed,
        "training_scaler_identifier": scaler_identifier,
        "n_train_endpoints": len(train.targets),
        "n_validation_endpoints": len(validation.targets),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "stopping_reason": stopping_reason,
        "validation_mae": restored_metrics["mae"],
        "validation_rmse": restored_metrics["rmse"],
        "validation_directional_accuracy": restored_metrics["directional_accuracy"],
        "actual_positive_rate": restored_metrics["actual_positive_rate"],
        "fit_runtime_seconds": runtime,
        "completion_status": "completed",
        "retry_status": retry_status,
        "warning_status": "none",
        "warning_messages": "",
        "error_status": "none",
        "error_type": "",
        "error_message": "",
        "candidate_checkpoint_identifier": checkpoint_identifier,
    }
    return CandidateFitResult(
        audit=audit,
        predictions=predictions,
        checkpoint_payload=checkpoint_payload,
        checkpoint_identifier=checkpoint_identifier,
        state_dict=best_state,
    )


def fit_candidate_with_retry(**kwargs: Any) -> CandidateFitResult:
    """Retry the identical fixed seed once only for a technical exception."""

    try:
        return fit_candidate(**kwargs, retry_status="not_retried")
    except Exception as first_error:
        try:
            return fit_candidate(**kwargs, retry_status=f"retried_once_after_{type(first_error).__name__}")
        except Exception as second_error:
            raise LSTMEvaluationError(
                "Candidate seed failed twice with the same fixed identity: "
                f"first={type(first_error).__name__}: {first_error}; "
                f"second={type(second_error).__name__}: {second_error}."
            ) from second_error


def candidate_results_frames(results: Sequence[CandidateFitResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audits = pd.DataFrame([result.audit for result in results], columns=CANDIDATE_AUDIT_COLUMNS)
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    return audits, predictions.loc[:, CANDIDATE_PREDICTION_COLUMNS]


def validate_complete_candidate_grid(audits: pd.DataFrame, predictions: pd.DataFrame) -> None:
    expected = {
        (fold, candidate.config_id, asset, seed)
        for fold in DEVELOPMENT_FOLD_IDS
        for candidate in FROZEN_CANDIDATES
        for asset in TARGET_ASSETS
        for seed in FROZEN_SEEDS
    }
    observed = set(
        audits[["fold", "model_config_id", "asset", "seed"]].itertuples(index=False, name=None)
    )
    if len(audits) != 120 or observed != expected or audits.duplicated(
        ["fold", "model_config_id", "asset", "seed"]
    ).any():
        raise LSTMEvaluationError("Candidate audit is not the complete 120-fit grid.")
    if set(audits["completion_status"]) != {"completed"}:
        raise LSTMEvaluationError("Candidate audit includes an incomplete fit.")
    if not np.isfinite(audits[["validation_mae", "validation_rmse"]].to_numpy()).all():
        raise LSTMEvaluationError("Candidate audit contains non-finite metrics.")

    key_columns = ["origin_date", "target_date"]
    for fold in DEVELOPMENT_FOLD_IDS:
        for asset in TARGET_ASSETS:
            reference: pd.MultiIndex | None = None
            for candidate in FROZEN_CANDIDATES:
                for seed in FROZEN_SEEDS:
                    part = predictions.loc[
                        (predictions["fold"] == fold)
                        & (predictions["asset"] == asset)
                        & (predictions["model_config_id"] == candidate.config_id)
                        & (predictions["seed"] == seed)
                    ]
                    if part.empty or part.duplicated(key_columns).any():
                        raise LSTMEvaluationError("Candidate validation slice is empty or duplicated.")
                    keys = pd.MultiIndex.from_frame(part.loc[:, key_columns])
                    if reference is None:
                        reference = keys
                    elif not keys.equals(reference):
                        raise LSTMEvaluationError(
                            f"Validation keys differ in {fold}/{asset} across candidates or seeds."
                        )


def select_fold_winners(audits: pd.DataFrame, predictions: pd.DataFrame, *, run_id: str) -> SelectionResult:
    validate_complete_candidate_grid(audits, predictions)
    rows: list[dict[str, Any]] = []
    winners: dict[str, LSTMCandidate] = {}
    for fold in DEVELOPMENT_FOLD_IDS:
        fold_rows: list[dict[str, Any]] = []
        for candidate in FROZEN_CANDIDATES:
            part = audits.loc[
                (audits["fold"] == fold) & (audits["model_config_id"] == candidate.config_id)
            ]
            by_asset = part.groupby("asset", sort=False)[["validation_mae", "validation_rmse"]].mean()
            if tuple(by_asset.index) != TARGET_ASSETS:
                by_asset = by_asset.reindex(TARGET_ASSETS)
            if by_asset.isna().any().any():
                raise LSTMEvaluationError(f"Incomplete candidate asset metrics for {fold}.")
            row: dict[str, Any] = {
                "run_id": run_id,
                "fold": fold,
                "model_config_id": candidate.config_id,
                "context_length": candidate.context_length,
                "macro_validation_mae": float(by_asset["validation_mae"].mean()),
                "macro_validation_rmse": float(by_asset["validation_rmse"].mean()),
            }
            for asset in TARGET_ASSETS:
                prefix = asset.lower()
                row[f"{prefix}_seed_mean_mae"] = float(by_asset.loc[asset, "validation_mae"])
                row[f"{prefix}_seed_mean_rmse"] = float(by_asset.loc[asset, "validation_rmse"])
            fold_rows.append(row)
        minimum = min(row["macro_validation_mae"] for row in fold_rows)
        tied = [row for row in fold_rows if abs(row["macro_validation_mae"] - minimum) <= MIN_DELTA]
        tied.sort(key=lambda row: (row["context_length"], row["macro_validation_rmse"], row["model_config_id"]))
        winner_id = tied[0]["model_config_id"]
        winner = next(candidate for candidate in FROZEN_CANDIDATES if candidate.config_id == winner_id)
        winners[fold] = winner
        ranked = sorted(fold_rows, key=lambda row: (row["macro_validation_mae"], row["model_config_id"]))
        ranks = {row["model_config_id"]: index + 1 for index, row in enumerate(ranked)}
        for row in fold_rows:
            row["primary_rank"] = ranks[row["model_config_id"]]
            row["within_practical_tie"] = abs(row["macro_validation_mae"] - minimum) <= MIN_DELTA
            row["selected"] = row["model_config_id"] == winner_id
            row["selection_reason"] = (
                "lowest_macro_validation_mae"
                if len(tied) == 1 and row["selected"]
                else "practical_tie_resolved_by_shorter_context_then_rmse_then_lexical"
                if row["selected"]
                else "not_selected"
            )
            rows.append(row)
    comparison = pd.DataFrame(rows, columns=SELECTION_COLUMNS).sort_values(
        ["fold", "model_config_id"], kind="mergesort", ignore_index=True
    )
    best_rows: list[dict[str, Any]] = []
    for fold, winner in winners.items():
        selected = audits.loc[
            (audits["fold"] == fold) & (audits["model_config_id"] == winner.config_id)
        ].sort_values(["asset", "seed"], kind="mergesort")
        for row in selected.itertuples(index=False):
            best_rows.append(
                {
                    "run_id": run_id,
                    "fold": fold,
                    "winning_model_config_id": winner.config_id,
                    "context_length": winner.context_length,
                    "asset": row.asset,
                    "seed": int(row.seed),
                    "best_epoch": int(row.best_epoch),
                    "candidate_validation_mae": float(row.validation_mae),
                    "candidate_checkpoint_identifier": row.candidate_checkpoint_identifier,
                }
            )
    best_epochs = pd.DataFrame(best_rows, columns=BEST_EPOCH_COLUMNS).sort_values(
        ["fold", "asset", "seed"], kind="mergesort", ignore_index=True
    )
    if len(best_epochs) != 60:
        raise LSTMEvaluationError("Winning best-epoch table must contain 60 rows.")
    return SelectionResult(comparison=comparison, best_epochs=best_epochs, winners=winners)


def refit_and_predict(
    refit: SequenceSet,
    test: SequenceSet,
    *,
    run_id: str,
    spec_version: str,
    data_sha256: str,
    model_config_sha256: str,
    fold: str,
    candidate: LSTMCandidate,
    asset: str,
    seed: int,
    epochs: int,
    scaler_identifier: str,
) -> RefitResult:
    started = perf_counter()
    seed_everything(seed)
    model = LSTMRegressor(context_length=candidate.context_length)
    losses = train_fixed_epochs(model, refit.features, refit.targets, epochs=epochs)
    if len(losses) != epochs:
        raise LSTMEvaluationError("Refit did not complete the requested epoch count.")
    test_predictions = predict(model, test.features)
    state = clone_state_dict(model)
    metadata = {
        "schema_version": 1,
        "stage": "winner_refit",
        "run_id": run_id,
        "fold": fold,
        "model_config_id": candidate.config_id,
        "context_length": candidate.context_length,
        "asset": asset,
        "seed": seed,
        "epochs": epochs,
        "feature_order": list(FROZEN_FEATURE_COLUMNS),
        "architecture": {
            "input_features": 17,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "bidirectional": False,
            "internal_dropout": 0.0,
            "post_lstm_dropout": POST_LSTM_DROPOUT,
            "output_size": OUTPUT_SIZE,
            "dtype": "float32",
        },
        "scaler_identifier": scaler_identifier,
        "processed_data_sha256": data_sha256,
        "model_search_sha256": model_config_sha256,
    }
    checkpoint_payload = _checkpoint_bytes(state, metadata)
    model_identifier = sha256_bytes(checkpoint_payload)
    rows = test.rows.copy()
    predictions = pd.DataFrame(
        {
            "run_id": run_id,
            "spec_version": spec_version,
            "data_version": data_sha256,
            "model": "lstm",
            "model_config_id": candidate.config_id,
            "seed": seed,
            "fold": fold,
            "partition": "test",
            "asset": asset,
            "origin_date": rows["origin_date"],
            "target_date": rows["target_date"],
            "actual_log_return": rows["actual_log_return"].astype(float),
            "predicted_log_return": test_predictions,
            "actual_direction": rows["actual_direction"].astype(np.int8),
            "predicted_direction": (test_predictions > 0).astype(np.int8),
            "trend_regime": REGIME_PLACEHOLDER,
            "volatility_regime": REGIME_PLACEHOLDER,
            "transition_regime": REGIME_PLACEHOLDER,
        }
    ).loc[:, CANONICAL_PREDICTION_FIELDS]
    audit = {
        "run_id": run_id,
        "fold": fold,
        "model_config_id": candidate.config_id,
        "context_length": candidate.context_length,
        "asset": asset,
        "seed": seed,
        "refit_scaler_identifier": scaler_identifier,
        "n_refit_endpoints": len(refit.targets),
        "n_test_endpoints": len(test.targets),
        "epochs_requested": epochs,
        "epochs_completed": len(losses),
        "early_stopping_used": False,
        "fit_runtime_seconds": perf_counter() - started,
        "completion_status": "completed",
        "warning_status": "none",
        "warning_messages": "",
        "error_status": "none",
        "error_type": "",
        "error_message": "",
        "refit_model_identifier": model_identifier,
    }
    return RefitResult(
        audit=audit,
        predictions=predictions,
        checkpoint_payload=checkpoint_payload,
        model_identifier=model_identifier,
        state_dict=state,
    )


def validate_baseline(baseline: pd.DataFrame, *, data_sha256: str) -> pd.DataFrame:
    if tuple(baseline.columns) != CANONICAL_PREDICTION_FIELDS:
        raise LSTMEvaluationError("Baseline does not use the canonical 18-column schema.")
    frame = baseline.copy()
    if len(frame) != 5032 or frame.duplicated(list(MATCH_KEYS)).any():
        raise LSTMEvaluationError("Canonical baseline must contain 5,032 unique D1-D5 keys.")
    if set(frame["fold"]) != set(DEVELOPMENT_FOLD_IDS) or set(frame["partition"]) != {"test"}:
        raise LSTMEvaluationError("Canonical baseline is not D1-D5 test-only.")
    if set(frame["asset"]) != set(TARGET_ASSETS) or set(frame["data_version"]) != {data_sha256}:
        raise LSTMEvaluationError("Canonical baseline identity differs from Phase 2F.")
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    if (frame["target_date"].dt.year == 2025).any():
        raise LSTMEvaluationError("Baseline unexpectedly exposes 2025.")
    return frame.sort_values(list(MATCH_KEYS), kind="mergesort", ignore_index=True)


def assert_exact_baseline_match(predictions: pd.DataFrame, baseline: pd.DataFrame) -> None:
    if len(predictions) != 15096 or set(predictions["seed"]) != set(FROZEN_SEEDS):
        raise LSTMEvaluationError("Canonical LSTM predictions must retain 15,096 three-seed rows.")
    baseline_keys = pd.MultiIndex.from_frame(baseline.loc[:, MATCH_KEYS])
    for seed in FROZEN_SEEDS:
        part = predictions.loc[predictions["seed"] == seed].sort_values(
            list(MATCH_KEYS), kind="mergesort", ignore_index=True
        )
        if len(part) != 5032 or part.duplicated(list(MATCH_KEYS)).any():
            raise LSTMEvaluationError(f"Seed {seed} does not contain 5,032 unique keys.")
        keys = pd.MultiIndex.from_frame(part.loc[:, MATCH_KEYS])
        if not keys.equals(baseline_keys):
            raise LSTMEvaluationError(f"Seed {seed} keys do not exactly match the baseline.")
        if not np.allclose(
            part["actual_log_return"].to_numpy(float),
            baseline["actual_log_return"].to_numpy(float),
            rtol=1e-12,
            atol=1e-15,
        ) or not np.array_equal(
            part["actual_direction"].to_numpy(np.int8),
            baseline["actual_direction"].to_numpy(np.int8),
        ):
            raise LSTMEvaluationError(f"Seed {seed} actual values differ from the baseline.")


def compute_per_seed_metrics(
    predictions: pd.DataFrame, baseline: pd.DataFrame, *, run_id: str
) -> pd.DataFrame:
    assert_exact_baseline_match(predictions, baseline)
    rows: list[dict[str, Any]] = []
    baseline_columns = [*MATCH_KEYS, "actual_log_return", "actual_direction", "predicted_log_return", "predicted_direction"]
    renamed = baseline.loc[:, baseline_columns].rename(
        columns={
            "actual_log_return": "baseline_actual_log_return",
            "actual_direction": "baseline_actual_direction",
            "predicted_log_return": "baseline_predicted_log_return",
            "predicted_direction": "baseline_predicted_direction",
        }
    )
    merged = predictions.merge(renamed, on=list(MATCH_KEYS), how="left", validate="many_to_one")
    for (fold, asset, seed), part in merged.groupby(["fold", "asset", "seed"], sort=True):
        actual = part["actual_log_return"].to_numpy(float)
        predicted = part["predicted_log_return"].to_numpy(float)
        core = regression_metrics(actual, predicted)
        baseline_mae = float(np.mean(np.abs(actual - part["baseline_predicted_log_return"].to_numpy(float))))
        if baseline_mae == 0:
            mae_skill = np.nan
        else:
            mae_skill = 1.0 - core["mae"] / baseline_mae
        baseline_da = float(
            np.mean(
                part["baseline_predicted_direction"].to_numpy(np.int8)
                == part["actual_direction"].to_numpy(np.int8)
            )
        )
        rows.append(
            {
                "run_id": run_id,
                "fold": fold,
                "asset": asset,
                "seed": int(seed),
                "model_config_id": part["model_config_id"].iloc[0],
                "n": len(part),
                **core,
                "baseline_mae": baseline_mae,
                "mae_skill": mae_skill,
                "baseline_directional_accuracy": baseline_da,
                "da_difference": core["directional_accuracy"] - baseline_da,
            }
        )
    result = pd.DataFrame(rows, columns=PER_SEED_METRIC_COLUMNS)
    if len(result) != 60:
        raise LSTMEvaluationError("Per-seed metric table must contain 60 rows.")
    return result.sort_values(["fold", "asset", "seed"], kind="mergesort", ignore_index=True)


def _metric_summary(part: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRIC_NAMES:
        values = part[metric].to_numpy(dtype=np.float64)
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=0))
        result[f"{metric}_min"] = float(np.min(values))
        result[f"{metric}_max"] = float(np.max(values))
    return result


def compute_asset_seed_summaries(metrics: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, asset), part in metrics.groupby(["fold", "asset"], sort=True):
        if set(part["seed"]) != set(FROZEN_SEEDS) or len(part) != 3:
            raise LSTMEvaluationError("Asset seed summary requires exactly three seed metrics.")
        rows.append(
            {
                "run_id": run_id,
                "fold": fold,
                "asset": asset,
                "model_config_id": part["model_config_id"].iloc[0],
                "n_seeds": 3,
                "n_per_seed": int(part["n"].iloc[0]),
                **_metric_summary(part),
            }
        )
    result = pd.DataFrame(rows, columns=ASSET_SEED_SUMMARY_COLUMNS)
    if len(result) != 20:
        raise LSTMEvaluationError("Asset seed summary must contain 20 groups.")
    return result.sort_values(["fold", "asset"], kind="mergesort", ignore_index=True)


def compute_macro_by_seed(metrics: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, seed), part in metrics.groupby(["fold", "seed"], sort=True):
        if set(part["asset"]) != set(TARGET_ASSETS) or len(part) != 4:
            raise LSTMEvaluationError("Macro metrics require four equal-weight asset rows.")
        rows.append(
            {
                "run_id": run_id,
                "fold": fold,
                "seed": int(seed),
                "model_config_id": part["model_config_id"].iloc[0],
                "n_assets": 4,
                "n_observations": int(part["n"].sum()),
                **{metric: float(part[metric].mean()) for metric in METRIC_NAMES},
            }
        )
    result = pd.DataFrame(rows, columns=MACRO_BY_SEED_COLUMNS)
    if len(result) != 15:
        raise LSTMEvaluationError("Macro-by-seed metrics must contain 15 rows.")
    return result.sort_values(["fold", "seed"], kind="mergesort", ignore_index=True)


def compute_macro_seed_summaries(macro: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, part in macro.groupby("fold", sort=True):
        if set(part["seed"]) != set(FROZEN_SEEDS) or len(part) != 3:
            raise LSTMEvaluationError("Macro seed summary requires three seed rows.")
        rows.append(
            {
                "run_id": run_id,
                "fold": fold,
                "model_config_id": part["model_config_id"].iloc[0],
                "n_seeds": 3,
                "n_assets": 4,
                "n_observations_per_seed": int(part["n_observations"].iloc[0]),
                **_metric_summary(part),
            }
        )
    result = pd.DataFrame(rows, columns=MACRO_SEED_SUMMARY_COLUMNS)
    if len(result) != 5:
        raise LSTMEvaluationError("Macro seed summary must contain five fold rows.")
    return result.sort_values("fold", kind="mergesort", ignore_index=True)


def compute_development_summary(
    asset_summaries: pd.DataFrame,
    macro_summaries: pd.DataFrame,
    *,
    run_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append(scope: str, asset: str, metric: str, values: np.ndarray) -> None:
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        rows.append(
            {
                "run_id": run_id,
                "scope": scope,
                "asset": asset,
                "metric": metric,
                "n_folds": len(values),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "input_statistic": "fold_seed_mean_metric",
                "interpretation": "descriptive_variability_not_confidence_interval",
            }
        )

    for asset in TARGET_ASSETS:
        part = asset_summaries.loc[asset_summaries["asset"] == asset]
        if len(part) != 5:
            raise LSTMEvaluationError(f"Development summary requires five folds for {asset}.")
        for metric in METRIC_NAMES:
            append("asset", asset, metric, part[f"{metric}_mean"].to_numpy(float))
    if len(macro_summaries) != 5:
        raise LSTMEvaluationError("Development macro summary requires five folds.")
    for metric in METRIC_NAMES:
        append("macro", "ALL_EQUAL_WEIGHT", metric, macro_summaries[f"{metric}_mean"].to_numpy(float))
    return pd.DataFrame(rows, columns=DEVELOPMENT_SUMMARY_COLUMNS).sort_values(
        ["scope", "asset", "metric"], kind="mergesort", ignore_index=True
    )


def validate_canonical_predictions(predictions: pd.DataFrame, winners: Mapping[str, LSTMCandidate]) -> None:
    if tuple(predictions.columns) != CANONICAL_PREDICTION_FIELDS:
        raise LSTMEvaluationError("LSTM predictions violate the canonical 18-column order.")
    if len(predictions) != 15096 or set(predictions["seed"]) != set(FROZEN_SEEDS):
        raise LSTMEvaluationError("LSTM predictions must contain 15,096 rows and three seeds.")
    if set(predictions["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise LSTMEvaluationError("LSTM predictions must contain exactly D1-D5.")
    if set(predictions["partition"]) != {"test"} or set(predictions["model"]) != {"lstm"}:
        raise LSTMEvaluationError("LSTM predictions must be test-only with model=lstm.")
    target_years = set(pd.to_datetime(predictions["target_date"], errors="raise").dt.year)
    if target_years != {2020, 2021, 2022, 2023, 2024}:
        raise LSTMEvaluationError("LSTM predictions must contain only 2020-2024 targets.")
    if not np.isfinite(predictions["predicted_log_return"].to_numpy(float)).all():
        raise LSTMEvaluationError("LSTM predictions contain non-finite values.")
    expected_direction = (predictions["predicted_log_return"].to_numpy(float) > 0).astype(np.int8)
    if not np.array_equal(expected_direction, predictions["predicted_direction"].to_numpy(np.int8)):
        raise LSTMEvaluationError("Predicted directions are not sign-derived.")
    for column in ("trend_regime", "volatility_regime", "transition_regime"):
        if set(predictions[column]) != {REGIME_PLACEHOLDER}:
            raise LSTMEvaluationError("A learned-model regime column is not the frozen sentinel.")
    for fold, winner in winners.items():
        if set(predictions.loc[predictions["fold"] == fold, "model_config_id"]) != {winner.config_id}:
            raise LSTMEvaluationError(f"Predictions for {fold} do not use its fold winner.")


def compare_metric_tables(left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str], values: Sequence[str]) -> None:
    a = left.sort_values(list(keys), kind="mergesort", ignore_index=True)
    b = right.sort_values(list(keys), kind="mergesort", ignore_index=True)
    if len(a) != len(b) or not a.loc[:, keys].equals(b.loc[:, keys]):
        raise LSTMEvaluationError("Independent metric verification found a key mismatch.")
    if not np.allclose(a.loc[:, values].to_numpy(float), b.loc[:, values].to_numpy(float), rtol=1e-12, atol=1e-15, equal_nan=True):
        raise LSTMEvaluationError("Independent metric verification found a numeric mismatch.")
