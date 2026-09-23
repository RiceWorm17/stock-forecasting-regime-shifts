"""Leakage-aware data, fit, and verification logic for the Phase 2H-B pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from src.models.patchtst_model import (
    BATCH_SIZE,
    D_MODEL,
    FEEDFORWARD_DIM,
    FROZEN_CANDIDATES,
    FROZEN_FEATURE_COLUMNS,
    GRADIENT_CLIP_NORM,
    HEAD_DROPOUT,
    LAYER_NORM_EPS,
    MAX_EPOCHS,
    MIN_DELTA,
    N_HEADS,
    NUM_ENCODER_LAYERS,
    PATCH_LENGTH,
    PATCH_STRIDE,
    PATIENCE,
    PILOT_SEEDS,
    TARGET_ASSETS,
    TRANSFORMER_DROPOUT,
    PatchTSTCandidate,
    PatchTSTContract,
    PatchTSTRegressor,
    build_optimizer,
    clone_state_dict,
    make_loader,
    predict,
    seed_everything,
    train_one_epoch,
)


D1_TRAIN_START = pd.Timestamp("2015-01-01")
D1_TRAIN_END = pd.Timestamp("2018-12-31")
D1_VALIDATION_START = pd.Timestamp("2019-01-01")
D1_VALIDATION_END = pd.Timestamp("2019-12-31")
PILOT_FOLD = "D1"
PILOT_PARTITION = "validation"
EXPECTED_SCALERS = 4
EXPECTED_CANDIDATE_FITS = 16
EXPECTED_REPLAY_EXECUTIONS = 1
EXPECTED_TOTAL_TRAINING_EXECUTIONS = 17
PILOT_WALL_TIME_CAP_SECONDS = 1800.0
FULL_RUNTIME_CAP_SECONDS = 21600.0
FULL_FIT_COUNT = 180
FULL_RUNTIME_SAFETY_FACTOR = 1.25
P5_RTOL = 1e-7
P5_ATOL = 1e-9

SAMPLE_KEYS: tuple[str, ...] = ("asset", "origin_date", "target_date")
PILOT_PREDICTION_COLUMNS: tuple[str, ...] = (
    "run_id",
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
)
FIT_AUDIT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "partition",
    "model_config_id",
    "context_length",
    "patch_count",
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
    "warning_status",
    "warning_messages",
    "error_status",
    "error_type",
    "error_message",
    "candidate_checkpoint_identifier",
)


class PatchTSTPilotError(RuntimeError):
    """Raised when a frozen pilot invariant is violated."""


@dataclass(frozen=True)
class SequenceSet:
    features: np.ndarray
    targets: np.ndarray
    rows: pd.DataFrame
    unavailable_keys: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class ScalerRecord:
    scaler: StandardScaler
    fold: str
    asset: str
    row_count: int
    identifier: str
    payload: bytes


@dataclass(frozen=True)
class CandidateFitResult:
    audit: dict[str, Any]
    predictions: pd.DataFrame
    epoch_history: pd.DataFrame
    checkpoint_payload: bytes
    checkpoint_identifier: str
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class WrittenArtifact:
    path: Path
    sha256: str
    rows: int | None = None


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


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
        raise PatchTSTPilotError(f"{name} must contain only boolean/0/1 values.")
    return numeric.astype(bool)


def validate_pilot_samples(samples: pd.DataFrame, *, contract: PatchTSTContract) -> pd.DataFrame:
    """Validate only D1 train/validation rows while retaining warm-up history."""

    required = set(SAMPLE_KEYS) | {
        "target_log_return",
        "target_direction",
        "core_evaluation_eligible",
        *contract.feature_columns,
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise PatchTSTPilotError(f"Processed samples are missing columns: {missing}.")
    if contract.feature_columns != FROZEN_FEATURE_COLUMNS or contract.assets != TARGET_ASSETS:
        raise PatchTSTPilotError("Contract feature or asset order differs from the frozen pilot.")
    frame = samples.copy()
    frame["asset"] = frame["asset"].astype(str)
    if not set(frame["asset"]).issubset(set(TARGET_ASSETS)):
        raise PatchTSTPilotError("Pilot samples contain a non-target asset.")
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    if frame[["origin_date", "target_date"]].isna().any().any():
        raise PatchTSTPilotError("Pilot samples contain missing dates.")
    if (frame["target_date"] <= frame["origin_date"]).any():
        raise PatchTSTPilotError("Every target date must follow its origin date.")
    if (frame["target_date"] < D1_TRAIN_START).any() or (
        frame["target_date"] > D1_VALIDATION_END
    ).any():
        raise PatchTSTPilotError("Pilot samples must be restricted to D1 train/validation dates.")
    if frame.duplicated(list(SAMPLE_KEYS), keep=False).any():
        raise PatchTSTPilotError("Pilot samples contain duplicate sample identities.")
    frame["core_evaluation_eligible"] = _boolean(
        frame["core_evaluation_eligible"], "core_evaluation_eligible"
    )
    target = pd.to_numeric(frame["target_log_return"], errors="coerce")
    direction = pd.to_numeric(frame["target_direction"], errors="coerce")
    eligible = frame["core_evaluation_eligible"]
    if not np.isfinite(target.loc[eligible]).all() or not direction.loc[eligible].isin([0, 1]).all():
        raise PatchTSTPilotError("Eligible D1 rows contain invalid targets.")
    finite_features = np.isfinite(
        frame.loc[eligible, contract.feature_columns].to_numpy(dtype=np.float64)
    ).all(axis=1)
    if not finite_features.all():
        raise PatchTSTPilotError("Eligible D1 rows contain non-finite features.")
    frame["target_log_return"] = target
    frame["target_direction"] = direction.astype("Int64")
    return frame.sort_values(["asset", "origin_date"], kind="mergesort", ignore_index=True)


def split_d1_train_validation(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = samples.loc[samples["core_evaluation_eligible"]].copy()
    train = eligible.loc[
        eligible["target_date"].between(D1_TRAIN_START, D1_TRAIN_END, inclusive="both")
    ].copy()
    validation = eligible.loc[
        eligible["target_date"].between(
            D1_VALIDATION_START, D1_VALIDATION_END, inclusive="both"
        )
    ].copy()
    if train.empty or validation.empty:
        raise PatchTSTPilotError("D1 train or validation partition is empty.")
    if set(train["target_date"].dt.year) - {2015, 2016, 2017, 2018}:
        raise PatchTSTPilotError("D1 training contains an unauthorized target year.")
    if set(validation["target_date"].dt.year) != {2019}:
        raise PatchTSTPilotError("D1 validation must contain target year 2019 only.")
    return train, validation


def _feature_row_hash(rows: pd.DataFrame, feature_columns: Sequence[str]) -> str:
    columns = ["asset", "origin_date", "target_date", *feature_columns]
    canonical = rows.loc[:, columns].sort_values(
        ["asset", "origin_date", "target_date"], kind="mergesort", ignore_index=True
    )
    return sha256_bytes(canonical_csv_bytes(canonical))


def fit_candidate_scaler(
    rows: pd.DataFrame,
    *,
    asset: str,
    feature_columns: Sequence[str],
    processed_data_sha256: str,
    patchtst_config_sha256: str,
) -> ScalerRecord:
    """Fit one asset scaler on unique D1 training feature rows only."""

    if tuple(feature_columns) != FROZEN_FEATURE_COLUMNS:
        raise ValueError("Scaler feature order differs from the frozen 17 columns.")
    if rows.empty or set(rows["asset"].astype(str)) != {asset}:
        raise PatchTSTPilotError(f"Scaler rows for D1/{asset} are invalid.")
    if rows.duplicated(list(SAMPLE_KEYS), keep=False).any():
        raise PatchTSTPilotError("Scaler input contains duplicate canonical feature rows.")
    if not rows["target_date"].between(D1_TRAIN_START, D1_TRAIN_END, inclusive="both").all():
        raise PatchTSTPilotError("Candidate scaler received a non-training row.")
    values = rows.loc[:, feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise PatchTSTPilotError("Scaler input contains non-finite values.")
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(values)
    expected_mean = values.mean(axis=0)
    expected_variance = values.var(axis=0, ddof=0)
    expected_scale = np.sqrt(expected_variance)
    expected_scale[expected_variance == 0.0] = 1.0
    if not np.allclose(scaler.mean_, expected_mean, rtol=1e-13, atol=1e-15):
        raise PatchTSTPilotError("StandardScaler mean differs from the row-level mean.")
    if not np.allclose(scaler.var_, expected_variance, rtol=1e-12, atol=1e-15):
        raise PatchTSTPilotError("StandardScaler variance is not population variance.")
    if not np.allclose(scaler.scale_, expected_scale, rtol=1e-12, atol=1e-15):
        raise PatchTSTPilotError("StandardScaler effective scale violates the frozen rule.")
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "patchtst_pilot_candidate_scaler",
        "model_family": "patchtst",
        "fold": PILOT_FOLD,
        "asset": asset,
        "usage": "candidate_shared_across_contexts_and_pilot_seeds",
        "fit_partition": "D1_train",
        "target_scaled": False,
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
            feature_columns[index]
            for index, value in enumerate(scaler.var_)
            if value == 0.0
        ],
        "canonical_feature_rows_sha256": _feature_row_hash(rows, feature_columns),
        "processed_data_sha256": processed_data_sha256,
        "patchtst_search_sha256": patchtst_config_sha256,
    }
    payload = canonical_json_bytes(metadata)
    return ScalerRecord(
        scaler=scaler,
        fold=PILOT_FOLD,
        asset=asset,
        row_count=len(rows),
        identifier=sha256_bytes(payload),
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
    """Build asset-local, unpadded windows ending exactly at each origin date."""

    if context_length not in {candidate.context_length for candidate in FROZEN_CANDIDATES}:
        raise ValueError("context_length must be 63 or 126.")
    if endpoints.empty:
        raise PatchTSTPilotError(f"{partition_name} endpoints are empty.")
    assets = set(endpoints["asset"].astype(str))
    if len(assets) != 1:
        raise PatchTSTPilotError("Sequence endpoints must contain exactly one asset.")
    asset = next(iter(assets))
    asset_history = history.loc[history["asset"].astype(str) == asset].copy()
    asset_history = asset_history.sort_values("origin_date", kind="mergesort", ignore_index=True)
    if asset_history["origin_date"].duplicated().any():
        raise PatchTSTPilotError(f"History for {asset} has duplicate origin dates.")
    if (asset_history["target_date"] > D1_VALIDATION_END).any():
        raise PatchTSTPilotError("Sequence history crossed the D1 validation boundary.")
    positions = {
        pd.Timestamp(value): index for index, value in enumerate(asset_history["origin_date"])
    }
    raw_features = asset_history.loc[:, FROZEN_FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    windows: list[np.ndarray] = []
    targets: list[float] = []
    row_records: list[dict[str, Any]] = []
    unavailable: list[tuple[str, str, str]] = []
    ordered = endpoints.sort_values("origin_date", kind="mergesort", ignore_index=True)
    for row in ordered.itertuples(index=False):
        origin = pd.Timestamp(row.origin_date)
        target_date = pd.Timestamp(row.target_date)
        key = (asset, origin.date().isoformat(), target_date.date().isoformat())
        end = positions.get(origin)
        start = -1 if end is None else end - context_length + 1
        if end is None or start < 0:
            unavailable.append(key)
            continue
        raw_window = raw_features[start : end + 1]
        if raw_window.shape != (context_length, len(FROZEN_FEATURE_COLUMNS)):
            unavailable.append(key)
            continue
        context_dates = asset_history.iloc[start : end + 1]["origin_date"]
        if context_dates.iloc[-1] != origin or (context_dates > origin).any():
            raise PatchTSTPilotError("Sequence does not end causally at its origin date.")
        if not np.isfinite(raw_window).all():
            unavailable.append(key)
            continue
        transformed = scaler.transform(raw_window)
        if not np.isfinite(transformed).all():
            raise PatchTSTPilotError("Scaler produced non-finite sequence values.")
        target = float(row.target_log_return)
        actual_direction = int(row.target_direction)
        if not np.isfinite(target) or actual_direction not in {0, 1}:
            raise PatchTSTPilotError("Sequence endpoint contains an invalid target.")
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
        raise PatchTSTPilotError(
            f"{partition_name} has {len(unavailable)} nonconstructible keys; first={unavailable[0]}."
        )
    if not windows:
        raise PatchTSTPilotError(f"No constructible {partition_name} sequences for {asset}.")
    return SequenceSet(
        features=np.stack(windows).astype(np.float32, copy=False),
        targets=np.asarray(targets, dtype=np.float32),
        rows=pd.DataFrame(row_records),
        unavailable_keys=tuple(unavailable),
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise PatchTSTPilotError("Metric arrays must be aligned finite vectors.")
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean(np.square(y - p)))),
        "directional_accuracy": float(
            np.mean((p > 0).astype(np.int8) == (y > 0).astype(np.int8))
        ),
        "actual_positive_rate": float(np.mean(y > 0)),
    }


def _checkpoint_bytes(
    state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "metadata": dict(metadata),
            "state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()
            },
        },
        buffer,
    )
    return buffer.getvalue()


def fit_candidate(
    train: SequenceSet,
    validation: SequenceSet,
    *,
    run_id: str,
    candidate: PatchTSTCandidate,
    asset: str,
    seed: int,
    scaler_identifier: str,
    processed_data_sha256: str,
    patchtst_config_sha256: str,
) -> CandidateFitResult:
    """Fit one frozen pilot candidate and restore its best validation-MAE state."""

    if candidate not in FROZEN_CANDIDATES:
        raise ValueError("Candidate is not one of the exact two frozen contexts.")
    if asset not in TARGET_ASSETS:
        raise ValueError("Asset is not one of the four pilot targets.")
    if int(seed) not in PILOT_SEEDS:
        raise ValueError("Pilot fit seed must be 1729 or 2718.")
    if train.features.shape[1] != candidate.context_length or (
        validation.features.shape[1] != candidate.context_length
    ):
        raise ValueError("Sequence context differs from the candidate context.")
    started = perf_counter()
    seed_everything(int(seed))
    warning_messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PatchTSTRegressor(context_length=candidate.context_length)
        optimizer = build_optimizer(model)
        loader = make_loader(train.features, train.targets)
        best_mae = np.inf
        best_epoch = 0
        best_metrics: dict[str, float] | None = None
        best_state: dict[str, torch.Tensor] | None = None
        non_improving = 0
        epochs_run = 0
        stopping_reason = "maximum_epochs"
        epoch_records: list[dict[str, Any]] = []
        validation_actual = validation.rows["actual_log_return"].to_numpy(dtype=np.float64)
        for epoch in range(1, MAX_EPOCHS + 1):
            train_mae = train_one_epoch(model, loader, optimizer)
            validation_predictions = predict(model, validation.features)
            metrics = regression_metrics(validation_actual, validation_predictions)
            epochs_run = epoch
            if metrics["mae"] < best_mae - MIN_DELTA:
                best_mae = metrics["mae"]
                best_epoch = epoch
                best_metrics = metrics
                best_state = clone_state_dict(model)
                non_improving = 0
            else:
                non_improving += 1
            epoch_records.append(
                {
                    "run_id": run_id,
                    "model_config_id": candidate.config_id,
                    "asset": asset,
                    "seed": int(seed),
                    "epoch": epoch,
                    "train_row_weighted_l1": train_mae,
                    "validation_mae": metrics["mae"],
                    "validation_rmse": metrics["rmse"],
                    "validation_directional_accuracy": metrics[
                        "directional_accuracy"
                    ],
                    "loss_prediction_gradient_parameter_checks": "PASS",
                }
            )
            if non_improving >= PATIENCE:
                stopping_reason = "patience_exhausted"
                break
        warning_messages = [str(item.message) for item in caught]
    if best_state is None or best_metrics is None or not 1 <= best_epoch <= MAX_EPOCHS:
        raise PatchTSTPilotError("Candidate fit did not retain a valid best epoch.")
    model.load_state_dict(best_state, strict=True)
    restored_predictions = predict(model, validation.features)
    restored_metrics = regression_metrics(validation_actual, restored_predictions)
    for metric in ("mae", "rmse", "directional_accuracy", "actual_positive_rate"):
        if not np.isclose(restored_metrics[metric], best_metrics[metric], rtol=0.0, atol=1e-15):
            raise PatchTSTPilotError("Restored best state does not reproduce its validation metrics.")

    checkpoint_metadata = {
        "schema_version": 1,
        "stage": "patchtst_pilot_candidate_best_epoch",
        "run_id": run_id,
        "fold": PILOT_FOLD,
        "partition": PILOT_PARTITION,
        "model_config_id": candidate.config_id,
        "context_length": candidate.context_length,
        "patch_start_indices": list(candidate.patch_start_indices),
        "asset": asset,
        "seed": int(seed),
        "best_epoch": best_epoch,
        "feature_order": list(FROZEN_FEATURE_COLUMNS),
        "architecture": {
            "patch_length": PATCH_LENGTH,
            "stride": PATCH_STRIDE,
            "d_model": D_MODEL,
            "heads": N_HEADS,
            "encoder_layers": NUM_ENCODER_LAYERS,
            "feedforward_dimension": FEEDFORWARD_DIM,
            "transformer_dropout": TRANSFORMER_DROPOUT,
            "head_dropout": HEAD_DROPOUT,
            "layer_norm_eps": LAYER_NORM_EPS,
            "head_input_dimension": candidate.head_input_dimension,
            "dtype": "float32",
        },
        "scaler_identifier": scaler_identifier,
        "processed_data_sha256": processed_data_sha256,
        "patchtst_search_sha256": patchtst_config_sha256,
    }
    checkpoint_payload = _checkpoint_bytes(best_state, checkpoint_metadata)
    checkpoint_identifier = sha256_bytes(checkpoint_payload)
    predictions = validation.rows.copy()
    predictions.insert(0, "asset", predictions.pop("asset"))
    predictions.insert(0, "partition", PILOT_PARTITION)
    predictions.insert(0, "fold", PILOT_FOLD)
    predictions.insert(0, "seed", int(seed))
    predictions.insert(0, "model_config_id", candidate.config_id)
    predictions.insert(0, "model", "patchtst")
    predictions.insert(0, "run_id", run_id)
    predictions["predicted_log_return"] = restored_predictions
    predictions["predicted_direction"] = (restored_predictions > 0).astype(np.int8)
    predictions = predictions.loc[:, PILOT_PREDICTION_COLUMNS]
    runtime = perf_counter() - started
    audit = {
        "run_id": run_id,
        "fold": PILOT_FOLD,
        "partition": PILOT_PARTITION,
        "model_config_id": candidate.config_id,
        "context_length": candidate.context_length,
        "patch_count": candidate.patch_count,
        "asset": asset,
        "seed": int(seed),
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
        "warning_status": "recorded" if warning_messages else "none",
        "warning_messages": " | ".join(warning_messages),
        "error_status": "none",
        "error_type": "",
        "error_message": "",
        "candidate_checkpoint_identifier": checkpoint_identifier,
    }
    return CandidateFitResult(
        audit=audit,
        predictions=predictions,
        epoch_history=pd.DataFrame(epoch_records),
        checkpoint_payload=checkpoint_payload,
        checkpoint_identifier=checkpoint_identifier,
        state_dict=best_state,
    )


def candidate_results_frames(
    results: Sequence[CandidateFitResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not results:
        raise PatchTSTPilotError("Candidate result collection is empty.")
    audits = pd.DataFrame([result.audit for result in results], columns=FIT_AUDIT_COLUMNS)
    predictions = pd.concat([result.predictions for result in results], ignore_index=True)
    predictions = predictions.sort_values(
        ["model_config_id", "asset", "seed", "origin_date", "target_date"],
        kind="mergesort",
        ignore_index=True,
    )
    return audits, predictions.loc[:, PILOT_PREDICTION_COLUMNS]


def expected_fit_identities() -> set[tuple[str, str, int]]:
    return {
        (candidate.config_id, asset, seed)
        for candidate in FROZEN_CANDIDATES
        for asset in TARGET_ASSETS
        for seed in PILOT_SEEDS
    }


def validate_complete_candidate_grid(audits: pd.DataFrame, predictions: pd.DataFrame) -> None:
    expected = expected_fit_identities()
    observed = set(
        audits[["model_config_id", "asset", "seed"]].itertuples(index=False, name=None)
    )
    if len(audits) != EXPECTED_CANDIDATE_FITS or observed != expected or audits.duplicated(
        ["model_config_id", "asset", "seed"]
    ).any():
        raise PatchTSTPilotError("Candidate audit is not the exact complete 16-fit grid.")
    if set(audits["completion_status"]) != {"completed"}:
        raise PatchTSTPilotError("Candidate audit includes an incomplete fit.")
    required_numeric = [
        "validation_mae",
        "validation_rmse",
        "validation_directional_accuracy",
        "actual_positive_rate",
        "fit_runtime_seconds",
    ]
    if not np.isfinite(audits[required_numeric].to_numpy(dtype=np.float64)).all():
        raise PatchTSTPilotError("Candidate audit contains a non-finite required value.")
    if set(predictions["fold"]) != {PILOT_FOLD} or set(predictions["partition"]) != {
        PILOT_PARTITION
    }:
        raise PatchTSTPilotError("Candidate predictions escaped D1 validation.")
    validation_coverage_audit(predictions)


def validation_coverage_audit(predictions: pd.DataFrame) -> dict[str, Any]:
    """Require exact D1 validation keys across both contexts and pilot seeds."""

    records: list[dict[str, Any]] = []
    for asset in TARGET_ASSETS:
        reference: pd.MultiIndex | None = None
        reference_count = 0
        for candidate in FROZEN_CANDIDATES:
            for seed in PILOT_SEEDS:
                part = predictions.loc[
                    (predictions["asset"] == asset)
                    & (predictions["model_config_id"] == candidate.config_id)
                    & (predictions["seed"] == seed)
                ].sort_values(["origin_date", "target_date"], kind="mergesort")
                if part.empty or part.duplicated(["origin_date", "target_date"]).any():
                    raise PatchTSTPilotError("Validation slice is empty or has duplicate keys.")
                keys = pd.MultiIndex.from_frame(part[["origin_date", "target_date"]])
                if reference is None:
                    reference = keys
                    reference_count = len(keys)
                elif not keys.equals(reference):
                    raise PatchTSTPilotError(
                        f"Validation keys differ for {asset} across contexts or seeds."
                    )
                records.append(
                    {
                        "asset": asset,
                        "model_config_id": candidate.config_id,
                        "seed": int(seed),
                        "validation_rows": len(part),
                        "key_sha256": sha256_bytes(
                            canonical_csv_bytes(part[["origin_date", "target_date"]])
                        ),
                    }
                )
        if reference_count == 0:
            raise PatchTSTPilotError(f"No validation reference keys for {asset}.")
    return {
        "schema_version": 1,
        "status": "PASS",
        "fold": PILOT_FOLD,
        "partition": PILOT_PARTITION,
        "no_intersection_used": True,
        "contexts": [candidate.config_id for candidate in FROZEN_CANDIDATES],
        "seeds": list(PILOT_SEEDS),
        "assets": list(TARGET_ASSETS),
        "slices": records,
    }


def verify_deterministic_replay(
    original: CandidateFitResult, replay: CandidateFitResult
) -> dict[str, Any]:
    """Apply the exact P5 discrete and allclose comparisons."""

    required_identity = ("PATCHTST_63", "AAPL", 1729)
    for result in (original, replay):
        identity = (
            str(result.audit["model_config_id"]),
            str(result.audit["asset"]),
            int(result.audit["seed"]),
        )
        if identity != required_identity:
            raise PatchTSTPilotError(f"P5 received incorrect replay identity: {identity}.")
    key_columns = ["asset", "origin_date", "target_date"]
    keys_equal = original.predictions[key_columns].equals(replay.predictions[key_columns])
    best_epoch_equal = int(original.audit["best_epoch"]) == int(replay.audit["best_epoch"])
    epochs_run_equal = int(original.audit["epochs_run"]) == int(replay.audit["epochs_run"])
    original_values = original.predictions["predicted_log_return"].to_numpy(dtype=np.float64)
    replay_values = replay.predictions["predicted_log_return"].to_numpy(dtype=np.float64)
    predictions_close = bool(
        np.allclose(original_values, replay_values, rtol=P5_RTOL, atol=P5_ATOL)
    )
    metric_fields = (
        "validation_mae",
        "validation_rmse",
        "validation_directional_accuracy",
    )
    metric_differences = {
        field: abs(float(original.audit[field]) - float(replay.audit[field]))
        for field in metric_fields
    }
    metrics_close = all(
        np.isclose(
            float(original.audit[field]),
            float(replay.audit[field]),
            rtol=P5_RTOL,
            atol=P5_ATOL,
        )
        for field in metric_fields
    )
    passed = keys_equal and best_epoch_equal and epochs_run_equal and predictions_close and metrics_close
    return {
        "schema_version": 1,
        "gate": "P5",
        "status": "PASS" if passed else "FAIL",
        "configuration": required_identity[0],
        "asset": required_identity[1],
        "seed": required_identity[2],
        "replay_is_candidate_grid_member": False,
        "keys_exact": keys_equal,
        "best_epoch_exact": best_epoch_equal,
        "epochs_run_exact": epochs_run_equal,
        "predictions_allclose": predictions_close,
        "metrics_allclose": metrics_close,
        "rtol": P5_RTOL,
        "atol": P5_ATOL,
        "max_abs_prediction_difference": float(np.max(np.abs(original_values - replay_values))),
        "metric_absolute_differences": metric_differences,
        "original": {
            "best_epoch": int(original.audit["best_epoch"]),
            "epochs_run": int(original.audit["epochs_run"]),
        },
        "replay": {
            "best_epoch": int(replay.audit["best_epoch"]),
            "epochs_run": int(replay.audit["epochs_run"]),
        },
    }


def compute_resource_feasibility(pilot_total_seconds: float) -> dict[str, Any]:
    """Apply P6 and the exact preregistered divisor-16 P7 formula."""

    total = float(pilot_total_seconds)
    if not np.isfinite(total) or total < 0.0:
        raise ValueError("pilot_total_seconds must be a finite nonnegative value.")
    implied = total / EXPECTED_CANDIDATE_FITS
    unadjusted = implied * FULL_FIT_COUNT
    estimate = unadjusted * FULL_RUNTIME_SAFETY_FACTOR
    return {
        "schema_version": 1,
        "pilot_total_seconds": total,
        "p6_cap_seconds": PILOT_WALL_TIME_CAP_SECONDS,
        "P6": "PASS" if total <= PILOT_WALL_TIME_CAP_SECONDS else "FAIL",
        "p7_denominator_candidate_fits": EXPECTED_CANDIDATE_FITS,
        "implied_seconds_per_denominator_fit": implied,
        "future_candidate_fits": 120,
        "future_refits": 60,
        "future_total_fits": FULL_FIT_COUNT,
        "unadjusted_180_fit_seconds": unadjusted,
        "safety_factor": FULL_RUNTIME_SAFETY_FACTOR,
        "estimated_full_seconds": estimate,
        "p7_cap_seconds": FULL_RUNTIME_CAP_SECONDS,
        "P7": "PASS" if estimate <= FULL_RUNTIME_CAP_SECONDS else "FAIL",
        "formula": "(pilot_total_seconds / 16) * 180 * 1.25",
    }


def build_gate_decision(
    *,
    integrity_passed: bool,
    grid_passed: bool,
    numerical_validity_passed: bool,
    coverage_passed: bool,
    replay_passed: bool,
    resource: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the technical-only P1-P7 decision with no performance criterion."""

    gates = {
        "P1": "PASS" if integrity_passed else "FAIL",
        "P2": "PASS" if grid_passed else "FAIL",
        "P3": "PASS" if numerical_validity_passed else "FAIL",
        "P4": "PASS" if coverage_passed else "FAIL",
        "P5": "PASS" if replay_passed else "FAIL",
        "P6": str(resource["P6"]),
        "P7": str(resource["P7"]),
    }
    if set(gates.values()) - {"PASS", "FAIL"}:
        raise PatchTSTPilotError("Every gate must have an unconditional PASS or FAIL status.")
    passed = all(value == "PASS" for value in gates.values())
    return {
        "schema_version": 1,
        "artifact_type": "phase2hb_patchtst_pilot_gate",
        "gates": gates,
        "decision": "PILOT_PASSED" if passed else "PILOT_NOT_PASSED",
        "performance_gate_used": False,
        "predictive_thresholds": [],
        "pilot_validation_metrics_are_diagnostic_only": True,
        "model_selection_performed": False,
        "full_benchmark_authorized": False,
        "resource": dict(resource),
    }


def independently_verify_saved_pilot(
    *,
    prediction_path: str | Path,
    metrics_path: str | Path,
    gate_path: str | Path,
) -> dict[str, Any]:
    """Reload immutable outputs and recompute identities, keys, directions, and metrics."""

    predictions = pd.read_csv(prediction_path)
    metrics = pd.read_csv(metrics_path)
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    if tuple(predictions.columns) != PILOT_PREDICTION_COLUMNS:
        raise PatchTSTPilotError("Saved pilot prediction schema drifted.")
    if tuple(metrics.columns) != FIT_AUDIT_COLUMNS:
        raise PatchTSTPilotError("Saved pilot metrics schema drifted.")
    predictions["origin_date"] = pd.to_datetime(predictions["origin_date"], errors="raise")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="raise")
    observed = set(
        metrics[["model_config_id", "asset", "seed"]].itertuples(index=False, name=None)
    )
    if len(metrics) != EXPECTED_CANDIDATE_FITS or observed != expected_fit_identities():
        raise PatchTSTPilotError("Saved metrics do not contain exactly 16 fit identities.")
    if set(predictions["model_config_id"]) != {item.config_id for item in FROZEN_CANDIDATES}:
        raise PatchTSTPilotError("Saved predictions contain an unexpected context.")
    if set(predictions["seed"].astype(int)) != set(PILOT_SEEDS):
        raise PatchTSTPilotError("Saved predictions contain an unexpected seed.")
    if set(predictions["asset"]) != set(TARGET_ASSETS):
        raise PatchTSTPilotError("Saved predictions contain an unexpected asset.")
    if set(predictions["model"]) != {"patchtst"} or set(predictions["fold"]) != {PILOT_FOLD}:
        raise PatchTSTPilotError("Saved predictions contain an unexpected model or fold.")
    if set(predictions["partition"]) != {PILOT_PARTITION}:
        raise PatchTSTPilotError("Saved predictions are not validation-only.")
    if set(predictions["target_date"].dt.year) != {2019}:
        raise PatchTSTPilotError("Saved predictions contain a non-2019 target.")
    numeric = predictions[["actual_log_return", "predicted_log_return"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise PatchTSTPilotError("Saved predictions contain non-finite values.")
    expected_directions = (predictions["predicted_log_return"].to_numpy(float) > 0).astype(np.int8)
    if not np.array_equal(expected_directions, predictions["predicted_direction"].to_numpy(np.int8)):
        raise PatchTSTPilotError("Saved predicted directions are not sign-derived.")
    if not np.array_equal(
        (predictions["actual_log_return"].to_numpy(float) > 0).astype(np.int8),
        predictions["actual_direction"].to_numpy(np.int8),
    ):
        raise PatchTSTPilotError("Saved actual directions do not match actual returns.")
    coverage = validation_coverage_audit(predictions)
    maximum_metric_difference = 0.0
    for identity in sorted(expected_fit_identities()):
        config_id, asset, seed = identity
        part = predictions.loc[
            (predictions["model_config_id"] == config_id)
            & (predictions["asset"] == asset)
            & (predictions["seed"] == seed)
        ]
        stored = metrics.loc[
            (metrics["model_config_id"] == config_id)
            & (metrics["asset"] == asset)
            & (metrics["seed"] == seed)
        ]
        if len(stored) != 1:
            raise PatchTSTPilotError(f"Saved metric row missing or duplicated for {identity}.")
        recomputed = regression_metrics(
            part["actual_log_return"].to_numpy(float),
            part["predicted_log_return"].to_numpy(float),
        )
        mapping = {
            "validation_mae": "mae",
            "validation_rmse": "rmse",
            "validation_directional_accuracy": "directional_accuracy",
            "actual_positive_rate": "actual_positive_rate",
        }
        for column, metric in mapping.items():
            difference = abs(float(stored.iloc[0][column]) - recomputed[metric])
            maximum_metric_difference = max(maximum_metric_difference, difference)
            if not np.isclose(
                float(stored.iloc[0][column]), recomputed[metric], rtol=1e-12, atol=1e-15
            ):
                raise PatchTSTPilotError(f"Saved metric {column} did not recompute for {identity}.")
    gate_keys = tuple(gate.get("gates", {}))
    if gate_keys != ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
        raise PatchTSTPilotError("Saved gate does not contain exactly P1 through P7.")
    if set(gate["gates"].values()) - {"PASS", "FAIL"}:
        raise PatchTSTPilotError("Saved gate contains a conditional or invalid status.")
    return {
        "schema_version": 1,
        "status": "PASS",
        "candidate_fit_identities": len(observed),
        "contexts": [item.config_id for item in FROZEN_CANDIDATES],
        "seeds": list(PILOT_SEEDS),
        "assets": list(TARGET_ASSETS),
        "folds": [PILOT_FOLD],
        "partitions": [PILOT_PARTITION],
        "target_years": [2019],
        "d1_test_or_2020_rows": 0,
        "d2_d5_rows": 0,
        "f1_or_2025_rows": 0,
        "finite_predictions": True,
        "directions_sign_derived": True,
        "validation_key_equality": coverage["status"] == "PASS",
        "saved_metrics_recomputed": True,
        "maximum_metric_absolute_difference": maximum_metric_difference,
        "gate_reloaded_from_disk": str(Path(gate_path)),
        "gate_decision": gate["decision"],
        "performance_gate_used": bool(gate["performance_gate_used"]),
    }
