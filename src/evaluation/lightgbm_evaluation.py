"""Leakage-aware D1--D5 evaluation for the frozen LightGBM benchmark.

The public orchestration functions in this module intentionally have no F1
switch.  They consume the preregistered model contract, fit candidates using
training/validation data only, refit fold winners, and compare development-test
predictions with the exact Phase 2C baseline keys.
"""

from __future__ import annotations

import hashlib
import io
import json
from time import perf_counter
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.schema import TARGET_ASSETS
from src.evaluation.walk_forward import (
    DEVELOPMENT_FOLD_IDS,
    FoldPartitions,
    FoldRegistry,
    load_fold_registry,
    split_samples_for_fold,
)
from src.models.lightgbm_model import (
    LightGBMCandidate,
    LightGBMContract,
    build_candidate_callbacks,
    build_lgbm_regressor,
    load_lightgbm_contract,
    validate_best_iteration,
)


MODEL_ID = "lightgbm"
SPEC_VERSION = "1.1"
LIGHTGBM_SEED = 1729
REGIME_PLACEHOLDER = "not_labeled_pre_regime_analysis"
BASELINE_MODEL_ID = "zero_return__direction_persistence"
BASELINE_CONFIG_ID = "zero_return__direction_persistence_v1"
BASELINE_REGIME_PLACEHOLDER = "not_labeled_phase2c"

SAMPLE_KEY_COLUMNS: tuple[str, ...] = (
    "fold",
    "asset",
    "origin_date",
    "target_date",
)
PREDICTION_COLUMNS: tuple[str, ...] = (
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
CANDIDATE_METRIC_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "fold",
    "model_config_id",
    "asset",
    "seed",
    "n_train",
    "n_validation",
    "best_iteration",
    "validation_mae",
    "validation_rmse",
    "validation_directional_accuracy",
    "actual_positive_count",
    "actual_nonpositive_count",
    "actual_positive_rate",
    "actual_nonpositive_rate",
    "fit_runtime_seconds",
    "status",
    "warning_status",
    "warning_count",
    "warning_messages",
    "error_status",
    "error_type",
    "error_message",
)
SELECTION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "fold",
    "model_config_id",
    "n_assets",
    "macro_validation_mae",
    "macro_validation_rmse",
    "delta_to_best_mae",
    "practically_tied",
    "simplicity_rank",
    "selected",
    "selection_reason",
)
PER_ASSET_METRIC_COLUMNS: tuple[str, ...] = (
    "run_id",
    "spec_version",
    "data_version",
    "model",
    "model_config_id",
    "seed",
    "fold",
    "asset",
    "n",
    "mae",
    "rmse",
    "directional_accuracy",
    "actual_positive_count",
    "actual_nonpositive_count",
    "actual_positive_rate",
    "actual_nonpositive_rate",
    "baseline_mae",
    "mae_skill",
    "baseline_directional_accuracy",
    "da_difference",
)
MACRO_METRIC_COLUMNS: tuple[str, ...] = (
    "run_id",
    "spec_version",
    "data_version",
    "model",
    "model_config_id",
    "seed",
    "fold",
    "n_assets",
    "n_observations",
    "mae",
    "rmse",
    "directional_accuracy",
    "actual_positive_rate",
    "actual_nonpositive_rate",
    "baseline_mae",
    "mae_skill",
    "baseline_directional_accuracy",
    "da_difference",
    "macro_weighting",
    "mae_skill_aggregation",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "scope",
    "asset",
    "metric",
    "n_folds",
    "n_defined",
    "median",
    "q1",
    "q3",
    "iqr",
    "quantile_interpolation",
    "interpretation",
)
BEST_ITERATION_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "fold",
    "model_config_id",
    "asset",
    "seed",
    "n_train",
    "n_validation",
    "best_iteration",
)
REFIT_AUDIT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "fold",
    "model_config_id",
    "asset",
    "seed",
    "n_train",
    "n_validation",
    "n_refit",
    "n_test",
    "n_estimators",
    "fit_runtime_seconds",
    "status",
    "warning_status",
    "warning_count",
    "warning_messages",
    "error_status",
    "error_type",
    "error_message",
)

_REQUIRED_NONFEATURE_COLUMNS: tuple[str, ...] = (
    "asset",
    "origin_date",
    "target_date",
    "target_log_return",
    "target_direction",
    "spy_observed",
    "core_evaluation_eligible",
)
_SUMMARY_METRICS: tuple[str, ...] = (
    "mae",
    "rmse",
    "directional_accuracy",
    "actual_positive_rate",
    "baseline_mae",
    "mae_skill",
    "baseline_directional_accuracy",
    "da_difference",
)


@dataclass(frozen=True)
class CandidateFitResult:
    """One train-only/validation candidate fit for one fold and asset."""

    fold: str
    candidate: LightGBMCandidate
    asset: str
    seed: int
    n_train: int
    n_validation: int
    best_iteration: int
    validation_mae: float
    validation_rmse: float
    validation_directional_accuracy: float
    actual_positive_count: int
    actual_nonpositive_count: int
    actual_positive_rate: float
    actual_nonpositive_rate: float
    fit_runtime_seconds: float
    status: str
    warning_status: str
    warning_messages: tuple[str, ...]
    error_status: str
    error_type: str
    error_message: str
    validation_predictions: pd.DataFrame = field(repr=False, compare=False)
    validation_history: tuple[float, ...] = field(repr=False, compare=False)
    estimator: Any = field(repr=False, compare=False)
    model_bytes: bytes = field(repr=False, compare=False)


@dataclass(frozen=True)
class FoldSelection:
    """Complete fold-local candidate ranking and its shared winner."""

    fold: str
    selected_config_id: str
    selected_macro_validation_mae: float
    selected_macro_validation_rmse: float
    best_iterations_by_asset: Mapping[str, int]
    table: pd.DataFrame = field(repr=False, compare=False)


@dataclass(frozen=True)
class RefitResult:
    """One winning-configuration train+validation refit and test forecast."""

    fold: str
    candidate: LightGBMCandidate
    asset: str
    seed: int
    n_train: int
    n_validation: int
    n_refit: int
    n_test: int
    n_estimators: int
    fit_runtime_seconds: float
    status: str
    warning_status: str
    warning_messages: tuple[str, ...]
    error_status: str
    error_type: str
    error_message: str
    predictions: pd.DataFrame = field(repr=False, compare=False)
    estimator: Any = field(repr=False, compare=False)
    model_bytes: bytes = field(repr=False, compare=False)


@dataclass(frozen=True)
class LightGBMDevelopmentRun:
    """All in-memory outputs from a complete D1--D5 run."""

    candidate_results: tuple[CandidateFitResult, ...]
    refit_results: tuple[RefitResult, ...]
    candidate_predictions: pd.DataFrame
    candidate_metrics: pd.DataFrame
    selections: pd.DataFrame
    best_iterations: pd.DataFrame
    refit_audit: pd.DataFrame
    predictions: pd.DataFrame
    per_asset_metrics: pd.DataFrame
    macro_metrics: pd.DataFrame
    development_summary: pd.DataFrame


@dataclass(frozen=True)
class WrittenArtifact:
    """One immutable artifact named by the SHA-256 of its bytes."""

    path: Path
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class LightGBMArtifacts:
    """Content-addressed tables and model states written for one run."""

    tables: Mapping[str, WrittenArtifact]
    candidate_models: tuple[WrittenArtifact, ...]
    refit_models: tuple[WrittenArtifact, ...]


def _validate_contract(contract: LightGBMContract) -> None:
    if contract.spec_version != SPEC_VERSION:
        raise ValueError(f"Experiment specification must be {SPEC_VERSION}.")
    if tuple(contract.assets) != TARGET_ASSETS:
        raise ValueError(f"Target assets must be exactly {TARGET_ASSETS}.")
    if tuple(contract.canonical_prediction_fields) != PREDICTION_COLUMNS:
        raise ValueError("Contract prediction fields differ from the canonical 18-field schema.")
    if contract.regime_placeholder != REGIME_PLACEHOLDER:
        raise ValueError("Contract regime placeholder differs from the frozen value.")
    if tuple(candidate.config_id for candidate in contract.candidates) != (
        "LGBM_01",
        "LGBM_02",
        "LGBM_03",
        "LGBM_04",
    ):
        raise ValueError("LightGBM candidate IDs differ from the frozen grid.")
    if len(contract.feature_columns) != 17 or len(set(contract.feature_columns)) != 17:
        raise ValueError("LightGBM requires exactly 17 unique ordered features.")
    if int(contract.fixed_parameters.get("random_seed", -1)) != LIGHTGBM_SEED:
        raise ValueError("LightGBM seed must be 1729.")
    if contract.max_candidate_fits != 80 or contract.max_winning_refits != 20:
        raise ValueError("LightGBM resource budget must remain 80 candidate fits plus 20 refits.")


def _as_boolean_flag(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values.")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="raise")
    if not numeric.isin([0, 1]).all():
        raise ValueError(f"{name} must contain only boolean or 0/1 values.")
    return numeric.astype(bool)


def validate_development_samples(
    samples: pd.DataFrame,
    *,
    contract: LightGBMContract | None = None,
) -> pd.DataFrame:
    """Normalize the frozen processed schema without inspecting model performance."""

    active_contract = contract or load_lightgbm_contract()
    _validate_contract(active_contract)
    if not isinstance(samples, pd.DataFrame):
        raise TypeError(f"samples must be a pandas DataFrame, got {type(samples).__name__}.")
    required = set(_REQUIRED_NONFEATURE_COLUMNS) | set(active_contract.feature_columns)
    missing = sorted(required.difference(samples.columns))
    if missing:
        raise ValueError(f"Processed samples are missing required columns: {missing}.")

    result = samples.copy()
    if result["asset"].isna().any():
        raise ValueError("asset must not contain missing values.")
    result["asset"] = result["asset"].astype("string")
    if set(result["asset"]) != set(active_contract.assets):
        raise ValueError("Processed samples must contain exactly the four frozen target assets.")
    for column in ("origin_date", "target_date"):
        result[column] = pd.to_datetime(result[column], errors="raise")
        if result[column].isna().any():
            raise ValueError(f"{column} must not contain missing values.")
        if result[column].dt.tz is not None:
            raise ValueError(f"{column} must be timezone-naive session dates.")
    if (result["target_date"] <= result["origin_date"]).any():
        raise ValueError("Every target_date must be strictly later than origin_date.")
    if result.duplicated(["asset", "origin_date", "target_date"], keep=False).any():
        raise ValueError("Processed samples contain duplicate sample identities.")

    result[active_contract.target_column] = pd.to_numeric(
        result[active_contract.target_column], errors="raise"
    ).astype("float64")
    if not np.isfinite(result[active_contract.target_column].to_numpy()).all():
        raise ValueError("target_log_return must contain only finite values.")
    target_direction = pd.to_numeric(result["target_direction"], errors="raise")
    if not target_direction.isin([0, 1]).all():
        raise ValueError("target_direction must contain only 0/1 values.")
    result["target_direction"] = target_direction.astype("int8")
    expected_direction = (result[active_contract.target_column] > 0).astype("int8")
    if not result["target_direction"].equals(expected_direction):
        raise ValueError("target_direction is inconsistent with target_log_return.")

    for column in active_contract.feature_columns:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    spy_observed = _as_boolean_flag(result["spy_observed"], "spy_observed")
    eligible = _as_boolean_flag(
        result["core_evaluation_eligible"], "core_evaluation_eligible"
    )
    finite = np.isfinite(
        result.loc[:, list(active_contract.feature_columns)].to_numpy(dtype="float64")
    ).all(axis=1)
    expected_eligible = finite & spy_observed.to_numpy()
    if not np.array_equal(eligible.to_numpy(), expected_eligible):
        raise ValueError(
            "core_evaluation_eligible must equal finite frozen features and spy_observed."
        )
    result["spy_observed"] = spy_observed
    result["core_evaluation_eligible"] = eligible
    return result.sort_values(
        ["asset", "origin_date", "target_date"], kind="mergesort", ignore_index=True
    )


def prepare_development_fold(
    samples: pd.DataFrame,
    fold_id: str,
    *,
    contract: LightGBMContract | None = None,
    registry: FoldRegistry | None = None,
) -> FoldPartitions:
    """Return finite eligible train/validation/test rows for one D1--D5 fold."""

    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise ValueError("LightGBM development is restricted to D1--D5; F1 is prohibited.")
    active_contract = contract or load_lightgbm_contract()
    normalized = validate_development_samples(samples, contract=active_contract)
    active_registry = registry or load_fold_registry()
    if active_registry.default_fold_ids() != DEVELOPMENT_FOLD_IDS:
        raise ValueError("Fold registry must expose exactly D1--D5 for development.")
    split = split_samples_for_fold(
        normalized, fold_id, registry=active_registry, include_test=True
    )

    filtered: dict[str, pd.DataFrame] = {}
    for name in ("train", "validation", "test"):
        frame = getattr(split, name)
        frame = frame.loc[frame["core_evaluation_eligible"]].copy()
        if set(frame["asset"]) != set(active_contract.assets):
            raise ValueError(f"Fold {fold_id} {name} does not contain all frozen assets.")
        if not np.isfinite(
            frame.loc[:, list(active_contract.feature_columns)].to_numpy(dtype="float64")
        ).all():
            raise ValueError(f"Fold {fold_id} {name} contains non-finite eligible features.")
        filtered[name] = frame.sort_values(
            ["asset", "target_date", "origin_date"], kind="mergesort", ignore_index=True
        )
    return FoldPartitions(
        fold=split.fold,
        train=filtered["train"],
        validation=filtered["validation"],
        test=filtered["test"],
    )


def build_canonical_predictions(
    samples: pd.DataFrame,
    predicted_log_return: Sequence[float] | np.ndarray,
    *,
    run_id: str,
    data_version: str,
    fold_id: str,
    partition: str,
    model_config_id: str,
    contract: LightGBMContract | None = None,
) -> pd.DataFrame:
    """Construct canonical LightGBM rows from already generated forecasts."""

    active_contract = contract or load_lightgbm_contract()
    _validate_contract(active_contract)
    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise ValueError("Canonical development predictions may contain only D1--D5.")
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be validation or test.")
    if not run_id or not data_version:
        raise ValueError("run_id and data_version must be non-empty strings.")
    candidate = active_contract.candidate(model_config_id)
    del candidate  # Existence in the frozen candidate grid is the required check.
    values = np.asarray(predicted_log_return, dtype="float64").reshape(-1)
    if len(values) != len(samples):
        raise ValueError("Prediction count does not match sample count.")
    if not np.isfinite(values).all():
        raise ValueError("LightGBM predictions must be finite.")

    frame = pd.DataFrame(index=samples.index)
    frame["run_id"] = run_id
    frame["spec_version"] = active_contract.spec_version
    frame["data_version"] = data_version
    frame["model"] = MODEL_ID
    frame["model_config_id"] = model_config_id
    frame["seed"] = LIGHTGBM_SEED
    frame["fold"] = fold_id
    frame["partition"] = partition
    frame["asset"] = samples["asset"].astype("string")
    frame["origin_date"] = pd.to_datetime(samples["origin_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(samples["target_date"], errors="raise")
    frame["actual_log_return"] = pd.to_numeric(
        samples[active_contract.target_column], errors="raise"
    ).astype("float64")
    frame["predicted_log_return"] = values
    frame["actual_direction"] = pd.to_numeric(
        samples["target_direction"], errors="raise"
    ).astype("int8")
    frame["predicted_direction"] = (values > 0.0).astype("int8")
    for column in ("trend_regime", "volatility_regime", "transition_regime"):
        frame[column] = active_contract.regime_placeholder
    result = frame.loc[:, list(PREDICTION_COLUMNS)].sort_values(
        ["fold", "asset", "target_date", "origin_date"],
        kind="mergesort",
        ignore_index=True,
    )
    if (result["target_date"].dt.year >= 2025).any():
        raise ValueError("F1/2025 predictions are prohibited during development.")
    if result.duplicated(["fold", "asset", "origin_date", "target_date"]).any():
        raise ValueError("Canonical prediction rows contain duplicate sample keys.")
    return result


def _asset_slice(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    result = frame.loc[frame["asset"] == asset].sort_values(
        ["target_date", "origin_date"], kind="mergesort", ignore_index=True
    )
    if result.empty:
        raise ValueError(f"No eligible rows exist for asset {asset}.")
    return result


def serialize_lgbm_model(estimator: Any, *, num_iteration: int) -> bytes:
    """Serialize a fitted Booster deterministically as UTF-8 model text."""

    booster = getattr(estimator, "booster_", None)
    if booster is None or not hasattr(booster, "model_to_string"):
        raise TypeError("Fitted estimator does not expose booster_.model_to_string().")
    text = booster.model_to_string(num_iteration=num_iteration)
    if not isinstance(text, str) or not text:
        raise ValueError("LightGBM model serialization returned no text.")
    return (text.rstrip("\r\n") + "\n").encode("utf-8")


def _validation_history_values(history: Mapping[str, Any]) -> tuple[float, ...]:
    dataset = history.get("validation")
    if not isinstance(dataset, Mapping):
        return ()
    values = dataset.get("l1")
    if not isinstance(values, Sequence):
        return ()
    return tuple(float(value) for value in values)


def fit_candidate_asset(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    fold_id: str,
    asset: str,
    candidate: LightGBMCandidate,
    run_id: str,
    data_version: str,
    contract: LightGBMContract | None = None,
    estimator_factory: Any | None = None,
    lightgbm_module: Any | None = None,
) -> CandidateFitResult:
    """Fit one asset candidate using validation only for early stopping."""

    active_contract = contract or load_lightgbm_contract()
    _validate_contract(active_contract)
    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise ValueError("Candidate fitting is restricted to D1--D5.")
    if asset not in active_contract.assets:
        raise ValueError(f"Unknown target asset {asset!r}.")
    if active_contract.candidate(candidate.config_id) != candidate:
        raise ValueError("Candidate differs from the frozen contract.")
    train_asset = _asset_slice(train, asset)
    validation_asset = _asset_slice(validation, asset)
    features = list(active_contract.feature_columns)
    x_train = train_asset.loc[:, features]
    y_train = train_asset[active_contract.target_column]
    x_validation = validation_asset.loc[:, features]
    y_validation = validation_asset[active_contract.target_column]

    estimator = build_lgbm_regressor(
        active_contract, candidate, estimator_factory=estimator_factory
    )
    history, callbacks = build_candidate_callbacks(
        active_contract, lightgbm_module=lightgbm_module
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        fit_started = perf_counter()
        estimator.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            eval_names=["validation"],
            eval_metric="l1",
            callbacks=list(callbacks),
        )
        fit_runtime_seconds = perf_counter() - fit_started
    warning_messages = tuple(
        f"{item.category.__name__}: {item.message}" for item in caught_warnings
    )
    max_estimators = int(active_contract.fixed_parameters["max_estimators"])
    best_iteration = validate_best_iteration(
        estimator,
        history,
        max_estimators=max_estimators,
        dataset_name="validation",
        metric_name="l1",
    )
    predicted = np.asarray(
        estimator.predict(x_validation, num_iteration=best_iteration), dtype="float64"
    ).reshape(-1)
    canonical = build_canonical_predictions(
        validation_asset,
        predicted,
        run_id=run_id,
        data_version=data_version,
        fold_id=fold_id,
        partition="validation",
        model_config_id=candidate.config_id,
        contract=active_contract,
    )
    errors = predicted - y_validation.to_numpy(dtype="float64")
    actual_direction = validation_asset["target_direction"].to_numpy(dtype="int8")
    predicted_direction = (predicted > 0.0).astype("int8")
    positive_count = int(actual_direction.sum())
    return CandidateFitResult(
        fold=fold_id,
        candidate=candidate,
        asset=asset,
        seed=LIGHTGBM_SEED,
        n_train=len(train_asset),
        n_validation=len(validation_asset),
        best_iteration=best_iteration,
        validation_mae=float(np.mean(np.abs(errors))),
        validation_rmse=float(np.sqrt(np.mean(np.square(errors)))),
        validation_directional_accuracy=float(
            np.mean(predicted_direction == actual_direction)
        ),
        actual_positive_count=positive_count,
        actual_nonpositive_count=len(actual_direction) - positive_count,
        actual_positive_rate=positive_count / len(actual_direction),
        actual_nonpositive_rate=(len(actual_direction) - positive_count)
        / len(actual_direction),
        fit_runtime_seconds=float(fit_runtime_seconds),
        status="completed",
        warning_status="recorded" if warning_messages else "none",
        warning_messages=warning_messages,
        error_status="none",
        error_type="",
        error_message="",
        validation_predictions=canonical,
        validation_history=_validation_history_values(history),
        estimator=estimator,
        model_bytes=serialize_lgbm_model(estimator, num_iteration=best_iteration),
    )


def candidate_results_to_frames(
    results: Sequence[CandidateFitResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert candidate results to deterministic prediction and metric tables."""

    if not results:
        raise ValueError("Candidate results must not be empty.")
    prediction_frame = pd.concat(
        [result.validation_predictions for result in results], ignore_index=True
    ).sort_values(
        ["fold", "model_config_id", "asset", "target_date", "origin_date"],
        kind="mergesort",
        ignore_index=True,
    )
    rows = [
        {
            "run_id": result.validation_predictions["run_id"].iloc[0],
            "model": MODEL_ID,
            "fold": result.fold,
            "model_config_id": result.candidate.config_id,
            "asset": result.asset,
            "seed": result.seed,
            "n_train": result.n_train,
            "n_validation": result.n_validation,
            "best_iteration": result.best_iteration,
            "validation_mae": result.validation_mae,
            "validation_rmse": result.validation_rmse,
            "validation_directional_accuracy": result.validation_directional_accuracy,
            "actual_positive_count": result.actual_positive_count,
            "actual_nonpositive_count": result.actual_nonpositive_count,
            "actual_positive_rate": result.actual_positive_rate,
            "actual_nonpositive_rate": result.actual_nonpositive_rate,
            "fit_runtime_seconds": result.fit_runtime_seconds,
            "status": result.status,
            "warning_status": result.warning_status,
            "warning_count": len(result.warning_messages),
            "warning_messages": json.dumps(
                result.warning_messages, ensure_ascii=False, separators=(",", ":")
            ),
            "error_status": result.error_status,
            "error_type": result.error_type,
            "error_message": result.error_message,
        }
        for result in results
    ]
    metrics = pd.DataFrame(rows, columns=CANDIDATE_METRIC_COLUMNS).sort_values(
        ["fold", "model_config_id", "asset"], kind="mergesort", ignore_index=True
    )
    return prediction_frame, metrics


def _identity_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, list(SAMPLE_KEY_COLUMNS)])


def validate_complete_candidate_grid(
    candidate_metrics: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
    expected_validation: pd.DataFrame,
    *,
    fold_id: str,
    contract: LightGBMContract | None = None,
) -> None:
    """Fail unless every candidate/asset validation slice is complete and identical."""

    active_contract = contract or load_lightgbm_contract()
    _validate_contract(active_contract)
    if tuple(candidate_metrics.columns) != CANDIDATE_METRIC_COLUMNS:
        raise ValueError("Candidate metric columns differ from the frozen schema.")
    if tuple(candidate_predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("Candidate prediction columns differ from the canonical schema.")
    metrics = candidate_metrics.loc[candidate_metrics["fold"] == fold_id]
    expected_pairs = {
        (candidate.config_id, asset)
        for candidate in active_contract.candidates
        for asset in active_contract.assets
    }
    observed_pairs = set(zip(metrics["model_config_id"], metrics["asset"], strict=False))
    if observed_pairs != expected_pairs or len(metrics) != len(expected_pairs):
        raise ValueError(f"Fold {fold_id} candidate grid is incomplete or duplicated.")
    if not metrics["status"].eq("completed").all():
        raise ValueError(f"Fold {fold_id} candidate grid contains failed fits.")
    if not metrics["error_status"].eq("none").all():
        raise ValueError(f"Fold {fold_id} candidate grid contains recorded errors.")
    if not metrics["seed"].eq(LIGHTGBM_SEED).all():
        raise ValueError("Candidate fits must all use seed 1729.")
    finite_columns = [
        "validation_mae",
        "validation_rmse",
        "validation_directional_accuracy",
        "actual_positive_rate",
        "actual_nonpositive_rate",
        "fit_runtime_seconds",
    ]
    if not np.isfinite(metrics[finite_columns]).all().all():
        raise ValueError("Candidate validation metrics must be finite.")
    if (metrics["fit_runtime_seconds"] < 0.0).any():
        raise ValueError("Candidate fit runtime must be non-negative.")
    if not metrics["validation_directional_accuracy"].between(0.0, 1.0).all():
        raise ValueError("Candidate validation directional accuracy must be in [0, 1].")
    if not np.allclose(
        metrics["actual_positive_rate"] + metrics["actual_nonpositive_rate"], 1.0
    ):
        raise ValueError("Candidate validation class-balance rates must sum to one.")

    prediction_subset = candidate_predictions.loc[candidate_predictions["fold"] == fold_id]
    for config_id, asset in sorted(expected_pairs):
        observed = prediction_subset.loc[
            (prediction_subset["model_config_id"] == config_id)
            & (prediction_subset["asset"] == asset)
        ].copy()
        expected = _asset_slice(expected_validation, asset).copy()
        if observed.duplicated(list(SAMPLE_KEY_COLUMNS)).any():
            raise ValueError(f"Duplicate validation keys for {fold_id}/{config_id}/{asset}.")
        expected_keys = pd.MultiIndex.from_frame(
            expected.assign(fold=fold_id).loc[:, list(SAMPLE_KEY_COLUMNS)]
        )
        if not _identity_index(observed).equals(expected_keys):
            if set(_identity_index(observed)) != set(expected_keys):
                raise ValueError(f"Validation key mismatch for {fold_id}/{config_id}/{asset}.")
        expected_actual = expected[active_contract.target_column].to_numpy(dtype="float64")
        expected_direction = expected["target_direction"].to_numpy(dtype="int8")
        if not np.array_equal(
            observed["actual_log_return"].to_numpy(dtype="float64"), expected_actual
        ):
            raise ValueError("Candidate validation actual returns differ from source rows.")
        if not np.array_equal(
            observed["actual_direction"].to_numpy(dtype="int8"), expected_direction
        ):
            raise ValueError("Candidate validation actual directions differ from source rows.")
        metric_row = metrics.loc[
            (metrics["model_config_id"] == config_id) & (metrics["asset"] == asset)
        ].iloc[0]
        if int(metric_row["n_validation"]) != len(observed):
            raise ValueError("Candidate validation row count disagrees with predictions.")
        errors = observed["predicted_log_return"].to_numpy() - observed[
            "actual_log_return"
        ].to_numpy()
        if not np.isclose(float(metric_row["validation_mae"]), np.mean(np.abs(errors))):
            raise ValueError("Candidate validation MAE disagrees with row predictions.")
        if not np.isclose(
            float(metric_row["validation_rmse"]), np.sqrt(np.mean(np.square(errors)))
        ):
            raise ValueError("Candidate validation RMSE disagrees with row predictions.")
        observed_da = float(
            np.mean(
                observed["predicted_direction"].to_numpy(dtype="int8")
                == observed["actual_direction"].to_numpy(dtype="int8")
            )
        )
        if not np.isclose(
            float(metric_row["validation_directional_accuracy"]), observed_da
        ):
            raise ValueError(
                "Candidate validation directional accuracy disagrees with predictions."
            )
        observed_positive_rate = float(observed["actual_direction"].mean())
        if not np.isclose(
            float(metric_row["actual_positive_rate"]), observed_positive_rate
        ):
            raise ValueError("Candidate validation class balance disagrees with predictions.")


def select_fold_winner(
    candidate_metrics: pd.DataFrame,
    *,
    fold_id: str,
    contract: LightGBMContract | None = None,
) -> FoldSelection:
    """Select one shared fold winner by macro validation MAE and frozen ties."""

    active_contract = contract or load_lightgbm_contract()
    _validate_contract(active_contract)
    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise ValueError("Winner selection is restricted to D1--D5.")
    if tuple(candidate_metrics.columns) != CANDIDATE_METRIC_COLUMNS:
        raise ValueError("Candidate metric columns differ from the frozen audit schema.")
    frame = candidate_metrics.loc[candidate_metrics["fold"] == fold_id].copy()
    expected = len(active_contract.candidates) * len(active_contract.assets)
    if len(frame) != expected or frame.duplicated(["model_config_id", "asset"]).any():
        raise ValueError(f"Fold {fold_id} requires one complete candidate/asset grid.")
    if not frame["status"].eq("completed").all():
        raise ValueError("Cannot select a winner from incomplete candidate fits.")
    if not frame["error_status"].eq("none").all():
        raise ValueError("Cannot select a winner from candidate fits with errors.")
    expected_ids = {candidate.config_id for candidate in active_contract.candidates}
    if set(frame["model_config_id"]) != expected_ids:
        raise ValueError("Candidate IDs differ from the frozen four-candidate grid.")
    for config_id, group in frame.groupby("model_config_id", sort=False):
        if set(group["asset"]) != set(active_contract.assets):
            raise ValueError(
                f"Candidate {config_id} does not contain the exact four target assets."
            )

    aggregates = (
        frame.groupby("model_config_id", sort=True, observed=True)
        .agg(
            n_assets=("asset", "nunique"),
            macro_validation_mae=("validation_mae", "mean"),
            macro_validation_rmse=("validation_rmse", "mean"),
        )
        .reset_index()
    )
    if not aggregates["n_assets"].eq(len(active_contract.assets)).all():
        raise ValueError("Every candidate must have all four equally weighted assets.")
    best_mae = float(aggregates["macro_validation_mae"].min())
    aggregates["delta_to_best_mae"] = (
        aggregates["macro_validation_mae"] - best_mae
    ).abs()
    aggregates["practically_tied"] = (
        aggregates["delta_to_best_mae"] <= active_contract.tie_tolerance
    )
    simplicity_order = sorted(
        active_contract.candidates,
        key=lambda item: (item.num_leaves, -item.min_child_samples, item.config_id),
    )
    simplicity_rank = {
        candidate.config_id: rank for rank, candidate in enumerate(simplicity_order, start=1)
    }
    aggregates["simplicity_rank"] = aggregates["model_config_id"].map(simplicity_rank)
    tied = aggregates.loc[aggregates["practically_tied"]].sort_values(
        ["simplicity_rank", "macro_validation_rmse", "model_config_id"],
        kind="mergesort",
    )
    winner = tied.iloc[0]
    selected_id = str(winner["model_config_id"])
    aggregates["selected"] = aggregates["model_config_id"].eq(selected_id)
    reason = "minimum_macro_validation_mae"
    if len(tied) > 1:
        reason = "practical_tie_resolved_by_simplicity_then_rmse_then_config_id"
    aggregates["selection_reason"] = np.where(aggregates["selected"], reason, "not_selected")
    run_ids = frame["run_id"].drop_duplicates()
    if len(run_ids) != 1:
        raise ValueError("Candidate grid must belong to exactly one run_id.")
    aggregates.insert(0, "fold", fold_id)
    aggregates.insert(0, "model", MODEL_ID)
    aggregates.insert(0, "run_id", run_ids.iloc[0])
    table = aggregates.loc[:, list(SELECTION_COLUMNS)].sort_values(
        ["fold", "model_config_id"], kind="mergesort", ignore_index=True
    )
    chosen_asset_rows = frame.loc[frame["model_config_id"] == selected_id]
    best_iterations = {
        asset: int(chosen_asset_rows.loc[chosen_asset_rows["asset"] == asset, "best_iteration"].iloc[0])
        for asset in active_contract.assets
    }
    return FoldSelection(
        fold=fold_id,
        selected_config_id=selected_id,
        selected_macro_validation_mae=float(winner["macro_validation_mae"]),
        selected_macro_validation_rmse=float(winner["macro_validation_rmse"]),
        best_iterations_by_asset=best_iterations,
        table=table,
    )


def refit_winner_asset_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    selection: FoldSelection,
    asset: str,
    run_id: str,
    data_version: str,
    contract: LightGBMContract | None = None,
    estimator_factory: Any | None = None,
) -> RefitResult:
    """Refit one asset winner on train+validation and predict frozen test rows."""

    active_contract = contract or load_lightgbm_contract()
    _validate_contract(active_contract)
    if selection.fold not in DEVELOPMENT_FOLD_IDS:
        raise ValueError("Refitting is restricted to D1--D5.")
    if asset not in active_contract.assets:
        raise ValueError(f"Unknown target asset {asset!r}.")
    if set(selection.best_iterations_by_asset) != set(active_contract.assets):
        raise ValueError("Selection must preserve one best iteration for every target asset.")
    candidate = active_contract.candidate(selection.selected_config_id)
    n_estimators = int(selection.best_iterations_by_asset[asset])
    max_estimators = int(active_contract.fixed_parameters["max_estimators"])
    if not 1 <= n_estimators <= max_estimators:
        raise ValueError("Selected best iteration is outside the frozen bounds.")
    train_asset = _asset_slice(train, asset)
    validation_asset = _asset_slice(validation, asset)
    refit = pd.concat([train_asset, validation_asset])
    refit = refit.sort_values(["target_date", "origin_date"], kind="mergesort")
    test_asset = _asset_slice(test, asset)
    features = list(active_contract.feature_columns)
    estimator = build_lgbm_regressor(
        active_contract,
        candidate,
        n_estimators=n_estimators,
        estimator_factory=estimator_factory,
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        fit_started = perf_counter()
        estimator.fit(refit.loc[:, features], refit[active_contract.target_column])
        fit_runtime_seconds = perf_counter() - fit_started
    warning_messages = tuple(
        f"{item.category.__name__}: {item.message}" for item in caught_warnings
    )
    predicted = np.asarray(
        estimator.predict(test_asset.loc[:, features], num_iteration=n_estimators),
        dtype="float64",
    ).reshape(-1)
    canonical = build_canonical_predictions(
        test_asset,
        predicted,
        run_id=run_id,
        data_version=data_version,
        fold_id=selection.fold,
        partition="test",
        model_config_id=candidate.config_id,
        contract=active_contract,
    )
    return RefitResult(
        fold=selection.fold,
        candidate=candidate,
        asset=asset,
        seed=LIGHTGBM_SEED,
        n_train=len(train_asset),
        n_validation=len(validation_asset),
        n_refit=len(refit),
        n_test=len(test_asset),
        n_estimators=n_estimators,
        fit_runtime_seconds=float(fit_runtime_seconds),
        status="completed",
        warning_status="recorded" if warning_messages else "none",
        warning_messages=warning_messages,
        error_status="none",
        error_type="",
        error_message="",
        predictions=canonical,
        estimator=estimator,
        model_bytes=serialize_lgbm_model(estimator, num_iteration=n_estimators),
    )


def validate_baseline_predictions(
    baseline_predictions: pd.DataFrame,
    *,
    data_version: str | None = None,
) -> pd.DataFrame:
    """Validate the canonical Phase 2C prediction artifact used as denominator."""

    if not isinstance(baseline_predictions, pd.DataFrame):
        raise TypeError("baseline_predictions must be a pandas DataFrame.")
    if tuple(baseline_predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("Baseline prediction columns differ from the canonical schema.")
    result = baseline_predictions.copy()
    for column in ("origin_date", "target_date"):
        result[column] = pd.to_datetime(result[column], errors="raise")
    for column in ("actual_log_return", "predicted_log_return"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    for column in ("actual_direction", "predicted_direction"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("int8")
        if not result[column].isin([0, 1]).all():
            raise ValueError(f"Baseline {column} must be binary.")
    if result.duplicated(list(SAMPLE_KEY_COLUMNS)).any():
        raise ValueError("Baseline predictions contain duplicate comparison keys.")
    if set(result["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise ValueError("Baseline predictions must contain exactly D1--D5.")
    if not result["partition"].eq("test").all():
        raise ValueError("Baseline comparison rows must all be test rows.")
    if not result["model"].eq(BASELINE_MODEL_ID).all():
        raise ValueError("Unexpected baseline model identifier.")
    if not result["model_config_id"].eq(BASELINE_CONFIG_ID).all():
        raise ValueError("Unexpected baseline configuration identifier.")
    if not result["predicted_log_return"].eq(0.0).all():
        raise ValueError("Canonical regression baseline must predict exactly zero.")
    if not result[list(("trend_regime", "volatility_regime", "transition_regime"))].eq(
        BASELINE_REGIME_PLACEHOLDER
    ).all().all():
        raise ValueError("Baseline regime fields differ from the Phase 2C sentinel.")
    if (result["target_date"].dt.year >= 2025).any():
        raise ValueError("Baseline comparison must not include F1/2025 rows.")
    if data_version is not None and not result["data_version"].eq(data_version).all():
        raise ValueError("Baseline data_version differs from the learned-model input.")
    if not np.isfinite(result[["actual_log_return", "predicted_log_return"]]).all().all():
        raise ValueError("Baseline return fields must be finite.")
    if not result["actual_direction"].equals(
        (result["actual_log_return"] > 0.0).astype("int8")
    ):
        raise ValueError("Baseline actual direction disagrees with actual return.")
    return result.sort_values(
        list(SAMPLE_KEY_COLUMNS), kind="mergesort", ignore_index=True
    )


def assert_exact_baseline_match(
    predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Return a one-to-one matched table or fail on any key/actual mismatch."""

    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("Learned prediction columns differ from the canonical schema.")
    learned = predictions.copy()
    for column in ("origin_date", "target_date"):
        learned[column] = pd.to_datetime(learned[column], errors="raise")
    if learned.duplicated(list(SAMPLE_KEY_COLUMNS)).any():
        raise ValueError("Learned predictions contain duplicate comparison keys.")
    if not learned["model"].eq(MODEL_ID).all() or not learned["partition"].eq("test").all():
        raise ValueError("Only LightGBM development-test predictions may be scored.")
    if set(learned["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise ValueError("Learned predictions must contain exactly D1--D5.")
    if not learned["seed"].eq(LIGHTGBM_SEED).all():
        raise ValueError("LightGBM predictions must use seed 1729.")
    if not learned[["trend_regime", "volatility_regime", "transition_regime"]].eq(
        REGIME_PLACEHOLDER
    ).all().all():
        raise ValueError("Learned-model regime fields must remain preregistration sentinels.")
    if (learned["target_date"].dt.year >= 2025).any():
        raise ValueError("F1/2025 predictions are forbidden in development metrics.")
    data_versions = learned["data_version"].drop_duplicates()
    if len(data_versions) != 1:
        raise ValueError("Learned predictions must have one data_version.")
    baseline = validate_baseline_predictions(
        baseline_predictions, data_version=str(data_versions.iloc[0])
    )
    learned_keys = set(_identity_index(learned))
    baseline_keys = set(_identity_index(baseline))
    if learned_keys != baseline_keys or len(learned) != len(baseline):
        missing = len(baseline_keys - learned_keys)
        extra = len(learned_keys - baseline_keys)
        raise ValueError(
            f"Learned/baseline key mismatch; missing={missing}, extra={extra}. "
            "Intersection-based shrinking is prohibited."
        )
    matched = learned.merge(
        baseline.loc[
            :,
            [
                *SAMPLE_KEY_COLUMNS,
                "actual_log_return",
                "predicted_log_return",
                "actual_direction",
                "predicted_direction",
            ],
        ],
        on=list(SAMPLE_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        suffixes=("_model", "_baseline"),
        sort=False,
    )
    if not np.allclose(
        matched["actual_log_return_model"].to_numpy(dtype="float64"),
        matched["actual_log_return_baseline"].to_numpy(dtype="float64"),
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "Learned and baseline actual returns differ beyond strict CSV round-trip tolerance."
        )
    if not np.array_equal(
        matched["actual_direction_model"].to_numpy(dtype="int8"),
        matched["actual_direction_baseline"].to_numpy(dtype="int8"),
    ):
        raise ValueError("Learned and baseline actual directions are not identical.")
    return matched.sort_values(
        list(SAMPLE_KEY_COLUMNS), kind="mergesort", ignore_index=True
    )


def compute_per_asset_metrics(
    predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute fold/asset metrics and exactly matched baseline-relative values."""

    matched = assert_exact_baseline_match(predictions, baseline_predictions)
    rows: list[dict[str, Any]] = []
    for (fold, asset), frame in matched.groupby(["fold", "asset"], sort=True):
        model_actual = frame["actual_log_return_model"].to_numpy(dtype="float64")
        model_predicted = frame["predicted_log_return_model"].to_numpy(dtype="float64")
        baseline_predicted = frame["predicted_log_return_baseline"].to_numpy(dtype="float64")
        actual_direction = frame["actual_direction_model"].to_numpy(dtype="int8")
        model_direction = frame["predicted_direction_model"].to_numpy(dtype="int8")
        baseline_direction = frame["predicted_direction_baseline"].to_numpy(dtype="int8")
        model_errors = model_predicted - model_actual
        baseline_errors = baseline_predicted - model_actual
        mae = float(np.mean(np.abs(model_errors)))
        baseline_mae = float(np.mean(np.abs(baseline_errors)))
        model_da = float(np.mean(model_direction == actual_direction))
        baseline_da = float(np.mean(baseline_direction == actual_direction))
        positive_count = int(actual_direction.sum())
        config_ids = frame["model_config_id"].drop_duplicates()
        if len(config_ids) != 1:
            raise ValueError(f"Fold {fold}/{asset} has multiple winning configuration IDs.")
        first = frame.iloc[0]
        rows.append(
            {
                "run_id": first["run_id"],
                "spec_version": first["spec_version"],
                "data_version": first["data_version"],
                "model": MODEL_ID,
                "model_config_id": config_ids.iloc[0],
                "seed": LIGHTGBM_SEED,
                "fold": fold,
                "asset": asset,
                "n": len(frame),
                "mae": mae,
                "rmse": float(np.sqrt(np.mean(np.square(model_errors)))),
                "directional_accuracy": model_da,
                "actual_positive_count": positive_count,
                "actual_nonpositive_count": len(frame) - positive_count,
                "actual_positive_rate": positive_count / len(frame),
                "actual_nonpositive_rate": (len(frame) - positive_count) / len(frame),
                "baseline_mae": baseline_mae,
                "mae_skill": np.nan if baseline_mae == 0.0 else 1.0 - mae / baseline_mae,
                "baseline_directional_accuracy": baseline_da,
                "da_difference": model_da - baseline_da,
            }
        )
    result = pd.DataFrame(rows, columns=PER_ASSET_METRIC_COLUMNS)
    return result.sort_values(["fold", "asset"], kind="mergesort", ignore_index=True)


def compute_macro_metrics(per_asset_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average asset-level metrics equally, including asset-level MAE skill."""

    if tuple(per_asset_metrics.columns) != PER_ASSET_METRIC_COLUMNS:
        raise ValueError("Per-asset metric columns differ from the frozen schema.")
    rows: list[dict[str, Any]] = []
    for fold, frame in per_asset_metrics.groupby("fold", sort=True):
        if set(frame["asset"]) != set(TARGET_ASSETS) or len(frame) != len(TARGET_ASSETS):
            raise ValueError(f"Fold {fold} must contain exactly one row per target asset.")
        config_ids = frame["model_config_id"].drop_duplicates()
        if len(config_ids) != 1:
            raise ValueError(f"Fold {fold} must use one shared structural winner.")
        skill = frame["mae_skill"].to_numpy(dtype="float64")
        first = frame.iloc[0]
        rows.append(
            {
                "run_id": first["run_id"],
                "spec_version": first["spec_version"],
                "data_version": first["data_version"],
                "model": MODEL_ID,
                "model_config_id": config_ids.iloc[0],
                "seed": LIGHTGBM_SEED,
                "fold": fold,
                "n_assets": len(frame),
                "n_observations": int(frame["n"].sum()),
                "mae": float(frame["mae"].mean()),
                "rmse": float(frame["rmse"].mean()),
                "directional_accuracy": float(frame["directional_accuracy"].mean()),
                "actual_positive_rate": float(frame["actual_positive_rate"].mean()),
                "actual_nonpositive_rate": float(frame["actual_nonpositive_rate"].mean()),
                "baseline_mae": float(frame["baseline_mae"].mean()),
                "mae_skill": float(np.mean(skill)) if np.isfinite(skill).all() else np.nan,
                "baseline_directional_accuracy": float(
                    frame["baseline_directional_accuracy"].mean()
                ),
                "da_difference": float(frame["da_difference"].mean()),
                "macro_weighting": "equal_weight_across_assets",
                "mae_skill_aggregation": "arithmetic_mean_of_asset_level_mae_skill",
            }
        )
    result = pd.DataFrame(rows, columns=MACRO_METRIC_COLUMNS)
    if set(result["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise ValueError("Macro metrics must contain exactly D1--D5.")
    return result.sort_values("fold", kind="mergesort", ignore_index=True)


def _summary_rows(frame: pd.DataFrame, *, scope: str, asset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first = frame.iloc[0]
    for metric in _SUMMARY_METRICS:
        values = frame[metric].to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        if len(finite) == len(values):
            q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75], method="linear")
        else:
            q1 = median = q3 = np.nan
        rows.append(
            {
                "run_id": first["run_id"],
                "model": MODEL_ID,
                "scope": scope,
                "asset": asset,
                "metric": metric,
                "n_folds": len(values),
                "n_defined": len(finite),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "quantile_interpolation": "linear",
                "interpretation": "descriptive_variability_not_formal_uncertainty",
            }
        )
    return rows


def compute_development_summary(
    per_asset_metrics: pd.DataFrame,
    macro_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Report median/IQR across D1--D5 for every asset and the macro series."""

    rows: list[dict[str, Any]] = []
    for asset in TARGET_ASSETS:
        frame = per_asset_metrics.loc[per_asset_metrics["asset"] == asset]
        if set(frame["fold"]) != set(DEVELOPMENT_FOLD_IDS):
            raise ValueError(f"Asset {asset} summary requires exactly D1--D5.")
        rows.extend(_summary_rows(frame, scope="asset", asset=asset))
    if set(macro_metrics["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise ValueError("Macro summary requires exactly D1--D5.")
    rows.extend(_summary_rows(macro_metrics, scope="macro", asset="equal_weight_macro"))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        ["scope", "asset", "metric"], kind="mergesort", ignore_index=True
    )


def build_winning_best_iteration_table(
    candidate_metrics: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    contract: LightGBMContract | None = None,
) -> pd.DataFrame:
    """Extract the 20 asset-specific iterations used by winner refits."""

    active_contract = contract or load_lightgbm_contract()
    _validate_contract(active_contract)
    if tuple(candidate_metrics.columns) != CANDIDATE_METRIC_COLUMNS:
        raise ValueError("Candidate metric columns differ from the frozen audit schema.")
    if tuple(selections.columns) != SELECTION_COLUMNS:
        raise ValueError("Selection columns differ from the frozen schema.")
    winners = selections.loc[selections["selected"]].copy()
    if len(winners) != len(DEVELOPMENT_FOLD_IDS):
        raise ValueError("Exactly one winning candidate is required for every D1--D5 fold.")
    if set(winners["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise ValueError("Winning candidates must cover exactly D1--D5.")
    chosen = candidate_metrics.merge(
        winners.loc[:, ["fold", "model_config_id"]],
        on=["fold", "model_config_id"],
        how="inner",
        validate="many_to_one",
    )
    if len(chosen) != active_contract.max_winning_refits:
        raise ValueError("Winning best-iteration table must contain exactly 20 rows.")
    for fold_id, frame in chosen.groupby("fold", sort=False):
        if set(frame["asset"]) != set(active_contract.assets) or len(frame) != len(
            active_contract.assets
        ):
            raise ValueError(f"Fold {fold_id} lacks one best iteration per target asset.")
    result = chosen.loc[:, list(BEST_ITERATION_COLUMNS)].copy()
    return result.sort_values(["fold", "asset"], kind="mergesort", ignore_index=True)


def refit_results_to_frame(results: Sequence[RefitResult]) -> pd.DataFrame:
    """Convert the 20 winner refits into a deterministic audit table."""

    if not results:
        raise ValueError("Refit results must not be empty.")
    rows = []
    for result in results:
        rows.append(
            {
                "run_id": result.predictions["run_id"].iloc[0],
                "model": MODEL_ID,
                "fold": result.fold,
                "model_config_id": result.candidate.config_id,
                "asset": result.asset,
                "seed": result.seed,
                "n_train": result.n_train,
                "n_validation": result.n_validation,
                "n_refit": result.n_refit,
                "n_test": result.n_test,
                "n_estimators": result.n_estimators,
                "fit_runtime_seconds": result.fit_runtime_seconds,
                "status": result.status,
                "warning_status": result.warning_status,
                "warning_count": len(result.warning_messages),
                "warning_messages": json.dumps(
                    result.warning_messages, ensure_ascii=False, separators=(",", ":")
                ),
                "error_status": result.error_status,
                "error_type": result.error_type,
                "error_message": result.error_message,
            }
        )
    frame = pd.DataFrame(rows, columns=REFIT_AUDIT_COLUMNS).sort_values(
        ["fold", "asset"], kind="mergesort", ignore_index=True
    )
    if len(frame) != 20 or frame.duplicated(["fold", "asset"]).any():
        raise ValueError("Winner-refit audit must contain 20 unique fold/asset rows.")
    if set(frame["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise ValueError("Winner refits must contain exactly D1--D5.")
    if not frame["status"].eq("completed").all() or not frame["error_status"].eq(
        "none"
    ).all():
        raise ValueError("Winner-refit audit contains an incomplete or failed fit.")
    if not (frame["n_train"] + frame["n_validation"]).eq(frame["n_refit"]).all():
        raise ValueError("Winner refit row counts must equal train plus validation rows.")
    return frame


def run_development_lightgbm(
    samples: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    *,
    run_id: str,
    data_version: str,
    contract: LightGBMContract | None = None,
    registry: FoldRegistry | None = None,
    estimator_factory: Any | None = None,
    lightgbm_module: Any | None = None,
    progress_callback: Any | None = None,
) -> LightGBMDevelopmentRun:
    """Execute the complete preregistered LightGBM D1--D5 procedure."""

    active_contract = contract or load_lightgbm_contract()
    _validate_contract(active_contract)
    normalized = validate_development_samples(samples, contract=active_contract)
    active_registry = registry or load_fold_registry()
    all_candidates: list[CandidateFitResult] = []
    all_refits: list[RefitResult] = []
    selection_tables: list[pd.DataFrame] = []

    candidate_completed = 0
    refit_completed = 0
    for fold_id in DEVELOPMENT_FOLD_IDS:
        split = prepare_development_fold(
            normalized, fold_id, contract=active_contract, registry=active_registry
        )
        fold_results: list[CandidateFitResult] = []
        for candidate in active_contract.candidates:
            for asset in active_contract.assets:
                result = fit_candidate_asset(
                    split.train,
                    split.validation,
                    fold_id=fold_id,
                    asset=asset,
                    candidate=candidate,
                    run_id=run_id,
                    data_version=data_version,
                    contract=active_contract,
                    estimator_factory=estimator_factory,
                    lightgbm_module=lightgbm_module,
                )
                fold_results.append(result)
                candidate_completed += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "candidate",
                            "completed": candidate_completed,
                            "total": active_contract.max_candidate_fits,
                            "fold": fold_id,
                            "model_config_id": candidate.config_id,
                            "asset": asset,
                        }
                    )
        fold_predictions, fold_metrics = candidate_results_to_frames(fold_results)
        validate_complete_candidate_grid(
            fold_metrics,
            fold_predictions,
            split.validation,
            fold_id=fold_id,
            contract=active_contract,
        )
        selection = select_fold_winner(
            fold_metrics, fold_id=fold_id, contract=active_contract
        )
        selection_tables.append(selection.table)
        for asset in active_contract.assets:
            refit_result = refit_winner_asset_and_predict(
                split.train,
                split.validation,
                split.test,
                selection=selection,
                asset=asset,
                run_id=run_id,
                data_version=data_version,
                contract=active_contract,
                estimator_factory=estimator_factory,
            )
            all_refits.append(refit_result)
            refit_completed += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "refit",
                        "completed": refit_completed,
                        "total": active_contract.max_winning_refits,
                        "fold": fold_id,
                        "model_config_id": selection.selected_config_id,
                        "asset": asset,
                    }
                )
        all_candidates.extend(fold_results)

    if len(all_candidates) != active_contract.max_candidate_fits:
        raise AssertionError("LightGBM candidate-fit budget count is incomplete.")
    if len(all_refits) != active_contract.max_winning_refits:
        raise AssertionError("LightGBM winning-refit count is incomplete.")
    candidate_predictions, candidate_metrics = candidate_results_to_frames(all_candidates)
    predictions = pd.concat(
        [result.predictions for result in all_refits], ignore_index=True
    ).sort_values(
        ["fold", "asset", "target_date", "origin_date"],
        kind="mergesort",
        ignore_index=True,
    )
    selections = pd.concat(selection_tables, ignore_index=True).sort_values(
        ["fold", "model_config_id"], kind="mergesort", ignore_index=True
    )
    best_iterations = build_winning_best_iteration_table(
        candidate_metrics, selections, contract=active_contract
    )
    refit_audit = refit_results_to_frame(all_refits)
    per_asset = compute_per_asset_metrics(predictions, baseline_predictions)
    macro = compute_macro_metrics(per_asset)
    summary = compute_development_summary(per_asset, macro)
    return LightGBMDevelopmentRun(
        candidate_results=tuple(all_candidates),
        refit_results=tuple(all_refits),
        candidate_predictions=candidate_predictions,
        candidate_metrics=candidate_metrics,
        selections=selections,
        best_iterations=best_iterations,
        refit_audit=refit_audit,
        predictions=predictions,
        per_asset_metrics=per_asset,
        macro_metrics=macro,
        development_summary=summary,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a table with stable dates, float precision, and LF endings."""

    serializable = frame.copy()
    for column in ("origin_date", "target_date"):
        if column in serializable:
            serializable[column] = pd.to_datetime(serializable[column], errors="raise").dt.strftime(
                "%Y-%m-%d"
            )
    stream = io.StringIO(newline="")
    serializable.to_csv(
        stream,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    return stream.getvalue().encode("utf-8")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_immutable(path: str | Path, payload: bytes) -> None:
    """Create bytes once, accepting only an already-identical artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if destination.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite immutable artifact {destination}.")


def write_content_addressed_bytes(
    directory: str | Path,
    *,
    prefix: str,
    suffix: str,
    payload: bytes,
) -> WrittenArtifact:
    """Write ``prefix_<sha256><suffix>`` and return its verified identity."""

    digest = sha256_bytes(payload)
    path = Path(directory) / f"{prefix}_{digest}{suffix}"
    write_immutable(path, payload)
    if sha256_bytes(path.read_bytes()) != digest:
        raise IOError(f"Post-write SHA-256 verification failed for {path}.")
    return WrittenArtifact(path=path, sha256=digest, byte_count=len(payload))


def write_development_artifacts(
    run: LightGBMDevelopmentRun,
    *,
    output_root: str | Path,
) -> LightGBMArtifacts:
    """Write all tables and fitted model text without creating a run manifest."""

    root = Path(output_root)
    table_specs = {
        "candidate_predictions": (
            root / "candidates" / "predictions",
            "lightgbm_validation_predictions",
            run.candidate_predictions,
        ),
        "candidate_metrics": (
            root / "candidates" / "metrics",
            "lightgbm_validation_by_asset",
            run.candidate_metrics,
        ),
        "selections": (
            root / "selections",
            "lightgbm_fold_candidates",
            run.selections,
        ),
        "best_iterations": (
            root / "selections",
            "lightgbm_winning_best_iterations",
            run.best_iterations,
        ),
        "refit_audit": (
            root / "refits",
            "lightgbm_winner_refit_audit",
            run.refit_audit,
        ),
        "predictions": (
            root / "predictions",
            "lightgbm_development_predictions",
            run.predictions,
        ),
        "per_asset_metrics": (
            root / "metrics",
            "lightgbm_metrics_by_asset",
            run.per_asset_metrics,
        ),
        "macro_metrics": (
            root / "metrics",
            "lightgbm_metrics_macro",
            run.macro_metrics,
        ),
        "development_summary": (
            root / "metrics",
            "lightgbm_development_summary",
            run.development_summary,
        ),
    }
    tables = {
        name: write_content_addressed_bytes(
            directory,
            prefix=prefix,
            suffix=".csv",
            payload=canonical_csv_bytes(frame),
        )
        for name, (directory, prefix, frame) in table_specs.items()
    }
    candidate_models = tuple(
        write_content_addressed_bytes(
            root
            / "models"
            / "candidate"
            / result.fold
            / result.candidate.config_id
            / result.asset,
            prefix="model",
            suffix=".txt",
            payload=result.model_bytes,
        )
        for result in run.candidate_results
    )
    refit_models = tuple(
        write_content_addressed_bytes(
            root / "models" / "refit" / result.fold / result.asset,
            prefix="model",
            suffix=".txt",
            payload=result.model_bytes,
        )
        for result in run.refit_results
    )
    return LightGBMArtifacts(
        tables=tables,
        candidate_models=candidate_models,
        refit_models=refit_models,
    )


__all__ = [
    "BEST_ITERATION_COLUMNS",
    "CANDIDATE_METRIC_COLUMNS",
    "MACRO_METRIC_COLUMNS",
    "PER_ASSET_METRIC_COLUMNS",
    "PREDICTION_COLUMNS",
    "REGIME_PLACEHOLDER",
    "REFIT_AUDIT_COLUMNS",
    "SELECTION_COLUMNS",
    "SUMMARY_COLUMNS",
    "CandidateFitResult",
    "FoldSelection",
    "LightGBMArtifacts",
    "LightGBMDevelopmentRun",
    "RefitResult",
    "WrittenArtifact",
    "assert_exact_baseline_match",
    "build_canonical_predictions",
    "build_winning_best_iteration_table",
    "candidate_results_to_frames",
    "canonical_csv_bytes",
    "canonical_json_bytes",
    "compute_development_summary",
    "compute_macro_metrics",
    "compute_per_asset_metrics",
    "fit_candidate_asset",
    "prepare_development_fold",
    "refit_winner_asset_and_predict",
    "refit_results_to_frame",
    "run_development_lightgbm",
    "select_fold_winner",
    "serialize_lgbm_model",
    "sha256_bytes",
    "validate_baseline_predictions",
    "validate_complete_candidate_grid",
    "validate_development_samples",
    "write_content_addressed_bytes",
    "write_development_artifacts",
    "write_immutable",
]
