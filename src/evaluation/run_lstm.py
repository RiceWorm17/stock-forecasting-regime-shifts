"""One-command execution of the frozen Phase 2F D1--D5 LSTM benchmark.

Allowed public commands::

    python -m src.evaluation.run_lstm --dry-run
    python -m src.evaluation.run_lstm

There is deliberately no fold selector, year override, or F1/final-test path.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.data.integrity import file_sha256
from src.evaluation.lstm_evaluation import (
    ASSET_SEED_SUMMARY_COLUMNS,
    BEST_EPOCH_COLUMNS,
    CANDIDATE_AUDIT_COLUMNS,
    CANDIDATE_PREDICTION_COLUMNS,
    DEVELOPMENT_SUMMARY_COLUMNS,
    MACRO_BY_SEED_COLUMNS,
    MACRO_SEED_SUMMARY_COLUMNS,
    MATCH_KEYS,
    METRIC_NAMES,
    PER_SEED_METRIC_COLUMNS,
    REFIT_AUDIT_COLUMNS,
    ScalerRecord,
    SequenceSet,
    WrittenArtifact,
    assert_exact_baseline_match,
    build_sequences,
    candidate_results_frames,
    canonical_csv_bytes,
    canonical_json_bytes,
    compare_metric_tables,
    compute_asset_seed_summaries,
    compute_development_summary,
    compute_macro_by_seed,
    compute_macro_seed_summaries,
    compute_per_seed_metrics,
    fit_candidate,
    fit_candidate_with_retry,
    fit_scaler_record,
    prepare_development_fold,
    refit_and_predict,
    select_fold_winners,
    sha256_bytes,
    validate_baseline,
    validate_canonical_predictions,
    validate_development_samples,
    write_content_addressed,
    write_immutable,
)
from src.evaluation.walk_forward import DEVELOPMENT_FOLD_IDS, load_fold_registry
from src.models.lstm_model import (
    CANONICAL_PREDICTION_FIELDS,
    FROZEN_CANDIDATES,
    FROZEN_FEATURE_COLUMNS,
    FROZEN_SEEDS,
    LSTMContract,
    LSTMCandidate,
    REGIME_PLACEHOLDER,
    TARGET_ASSETS,
    configure_cpu_runtime,
    load_lstm_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "lstm"

RAW_SHA256 = "55f74f058493102387f1c2d0a848bd673a3cfa0473d617dfdb34aea8a94f7736"
PROCESSED_SHA256 = "9352da6b434011f9844ef0acc1813ea33dce39152712752bba21a3005aabf7be"
BASELINE_SHA256 = "e09e4909e9af7b7836c5e0b2c8d7b320d2848d5da9b33b8c9492421e0a108bb4"
PREREGISTRATION_SHA256 = "be04b9ce8457c5b26a8042f548eccce8ddb4b5d1b86a1404a43ef1bb1a52d613"
MODEL_SEARCH_SHA256 = "e15e42ff2b2f4f2bc7ccc185195c26fd35851b259bc28e08a6ba2149b6aaf38c"
PYPROJECT_SHA256 = "7fc8cf9c6fbc6440061cc68f3a515df28b521c724b32f345a5447f7844226309"
ENVIRONMENT_SHA256 = "8abe2cdd73cc852b7645a80855f82afff34cedcda28945604832cda75382d64b"
LIGHTGBM_PREDICTION_SHA256 = "36445836777cba815bc240c0c7229838cdc2177e94f50bac7905b8e2a9ace0c7"
LIGHTGBM_MANIFEST_SHA256 = "6b90c0907a42933b3d184bc87a6bc720dbf846a9d751e475c2657539e815bf75"
LIGHTGBM_TREE_SHA256 = "6f1079692067a8ed49dcce3d3db52efd22e00883084575a3a804a6f942e9d970"
LIGHTGBM_TREE_FILE_COUNT = 113
EXACT_TORCH_RUNTIME_VERSION = "2.14.0+cpu"
EXACT_TORCH_DISTRIBUTION_VERSION = "2.14.0"
EXACT_SKLEARN_VERSION = "1.9.1"
EXPECTED_BASELINE_ROWS = 5_032
EXPECTED_PREDICTION_ROWS = 15_096
EXPECTED_CANDIDATE_FITS = 120
EXPECTED_REFITS = 60
RUN_ID = f"phase2f_lstm_dev_{PROCESSED_SHA256[:12]}"

RAW_DATA_PATH = REPOSITORY_ROOT / "data" / "raw" / "yfinance_daily_2015-01-01_2025-12-31_20260913T152313Z_55f74f058493.csv"
PROCESSED_DATA_PATH = REPOSITORY_ROOT / "data" / "processed" / f"supervised_v1_{PROCESSED_SHA256}.csv"
BASELINE_PATH = REPOSITORY_ROOT / "results" / "baselines" / "predictions" / f"baseline_predictions_{BASELINE_SHA256}.csv"
PREREGISTRATION_PATH = REPOSITORY_ROOT / "docs" / "MODEL_PREREGISTRATION.md"
MODEL_SEARCH_PATH = REPOSITORY_ROOT / "configs" / "model_search.yaml"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
ENVIRONMENT_PATH = OUTPUT_ROOT / "environment" / f"lstm_environment_{ENVIRONMENT_SHA256}.json"
LIGHTGBM_PREDICTION_PATH = REPOSITORY_ROOT / "results" / "lightgbm" / "predictions" / f"lightgbm_development_predictions_{LIGHTGBM_PREDICTION_SHA256}.csv"
LIGHTGBM_MANIFEST_PATH = REPOSITORY_ROOT / "results" / "lightgbm" / f"run_manifest_{LIGHTGBM_MANIFEST_SHA256}.json"


class Phase2FExecutionError(RuntimeError):
    """Raised when a frozen Phase 2F execution or acceptance gate fails."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _require_hash(path: Path, expected: str, label: str) -> str:
    observed = file_sha256(path)
    if observed != expected:
        raise Phase2FExecutionError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}."
        )
    return observed


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase2FExecutionError(f"Unable to load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Phase2FExecutionError(f"JSON root must be an object: {path}.")
    return value


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise Phase2FExecutionError(f"Required distribution {name!r} is unavailable.") from exc


def _lightgbm_tree_identity() -> dict[str, Any]:
    root = REPOSITORY_ROOT / "results" / "lightgbm"
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
    )
    records = [
        f"{path.relative_to(REPOSITORY_ROOT).as_posix()}\t{file_sha256(path)}"
        for path in files
    ]
    digest = hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()
    return {"file_count": len(files), "sha256": digest}


def _source_tree_identity() -> dict[str, Any]:
    files = sorted(
        (path for path in (REPOSITORY_ROOT / "src").rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
    )
    aggregate = hashlib.sha256()
    per_file: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        digest = file_sha256(path)
        per_file[relative] = digest
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "method": "sha256_of_sorted_relative_path_nul_file_sha256_lf_records",
        "sha256": aggregate.hexdigest(),
        "file_count": len(files),
        "files": per_file,
        "git_revision": None,
        "git_status": "not_available_repository_has_no_git_metadata",
    }


def verify_exact_environment() -> dict[str, Any]:
    """Verify the immutable lock and freeze CPU threads before any training."""

    _require_hash(ENVIRONMENT_PATH, ENVIRONMENT_SHA256, "LSTM environment manifest")
    manifest = _load_json(ENVIRONMENT_PATH)
    runtime_policy = configure_cpu_runtime()
    observed_packages = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": _distribution_version("PyYAML"),
        "scikit_learn": _distribution_version("scikit-learn"),
        "torch_distribution": _distribution_version("torch"),
        "torch_runtime": torch.__version__,
    }
    if manifest.get("packages") != observed_packages:
        raise Phase2FExecutionError(
            f"Phase 2F package versions drifted: {observed_packages!r}."
        )
    if torch.__version__ != EXACT_TORCH_RUNTIME_VERSION or _distribution_version("torch") != EXACT_TORCH_DISTRIBUTION_VERSION:
        raise Phase2FExecutionError("PyTorch differs from the frozen exact CPU build.")
    if _distribution_version("scikit-learn") != EXACT_SKLEARN_VERSION:
        raise Phase2FExecutionError("scikit-learn differs from the frozen exact version.")
    python_record = manifest.get("python", {})
    if (
        python_record.get("version") != platform.python_version()
        or Path(str(python_record.get("executable"))).resolve() != Path(sys.executable).resolve()
        or python_record.get("compiler") != platform.python_compiler()
    ):
        raise Phase2FExecutionError("Python runtime differs from the frozen environment.")
    expected_repository_inputs = {
        "model_preregistration_path": "docs/MODEL_PREREGISTRATION.md",
        "model_preregistration_sha256": PREREGISTRATION_SHA256,
        "model_search_path": "configs/model_search.yaml",
        "model_search_sha256": MODEL_SEARCH_SHA256,
        "pyproject_path": "pyproject.toml",
        "pyproject_sha256": PYPROJECT_SHA256,
    }
    if manifest.get("repository_inputs") != expected_repository_inputs:
        raise Phase2FExecutionError("Environment repository input identities drifted.")
    if manifest.get("device_policy") != {
        "cuda_available": False,
        "cuda_version": None,
        "device": "cpu",
        "policy": "cpu_only_reference_run",
    }:
        raise Phase2FExecutionError("Environment device policy is not frozen CPU-only.")
    if runtime_policy != {
        "device": "cpu",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "deterministic_algorithms_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }:
        raise Phase2FExecutionError(f"PyTorch runtime policy drifted: {runtime_policy!r}.")
    return {
        "manifest": manifest,
        "manifest_path": _relative(ENVIRONMENT_PATH),
        "manifest_sha256": ENVIRONMENT_SHA256,
        "packages": observed_packages,
        "runtime_policy": runtime_policy,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
    }


def verify_frozen_inputs(contract: LSTMContract) -> dict[str, Any]:
    """Verify every frozen identity and exclude 2025 before model validation."""

    _require_hash(RAW_DATA_PATH, RAW_SHA256, "Raw dataset")
    _require_hash(PROCESSED_DATA_PATH, PROCESSED_SHA256, "Processed dataset")
    _require_hash(BASELINE_PATH, BASELINE_SHA256, "Canonical baseline")
    _require_hash(PREREGISTRATION_PATH, PREREGISTRATION_SHA256, "Model preregistration")
    _require_hash(MODEL_SEARCH_PATH, MODEL_SEARCH_SHA256, "Model-search configuration")
    _require_hash(PYPROJECT_PATH, PYPROJECT_SHA256, "Phase 2F pyproject")
    _require_hash(LIGHTGBM_PREDICTION_PATH, LIGHTGBM_PREDICTION_SHA256, "Frozen LightGBM predictions")
    _require_hash(LIGHTGBM_MANIFEST_PATH, LIGHTGBM_MANIFEST_SHA256, "Frozen LightGBM manifest")
    tree = _lightgbm_tree_identity()
    if tree != {"file_count": LIGHTGBM_TREE_FILE_COUNT, "sha256": LIGHTGBM_TREE_SHA256}:
        raise Phase2FExecutionError(f"Frozen LightGBM artifact tree drifted: {tree!r}.")
    if contract.raw_sha256 != RAW_SHA256 or contract.processed_sha256 != PROCESSED_SHA256:
        raise Phase2FExecutionError("LSTM contract contains unexpected data identities.")
    if contract.feature_columns != FROZEN_FEATURE_COLUMNS or len(contract.feature_columns) != 17:
        raise Phase2FExecutionError("The frozen predictive feature order is not exactly 17 columns.")
    if contract.assets != TARGET_ASSETS or contract.candidates != FROZEN_CANDIDATES or contract.seeds != FROZEN_SEEDS:
        raise Phase2FExecutionError("Frozen assets, candidates, or seeds have drifted.")

    all_samples = pd.read_csv(PROCESSED_DATA_PATH)
    target_dates = pd.to_datetime(all_samples["target_date"], errors="raise")
    development_mask = target_dates.dt.year <= 2024
    if int((~development_mask).sum()) != 1000:
        raise Phase2FExecutionError("Expected exactly 1,000 mechanically retained final-year rows.")
    development = all_samples.loc[development_mask].copy()
    development["target_date"] = target_dates.loc[development_mask]
    baseline = validate_baseline(pd.read_csv(BASELINE_PATH), data_sha256=PROCESSED_SHA256)
    return {
        "samples": development,
        "baseline": baseline,
        "final_year_rows_excluded_before_validation": int((~development_mask).sum()),
        "lightgbm_tree": tree,
    }


def _asset_rows(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    result = frame.loc[frame["asset"] == asset].sort_values("origin_date", kind="mergesort", ignore_index=True)
    if result.empty:
        raise Phase2FExecutionError(f"Missing required asset slice {asset}.")
    return result


def _sequence_keys(sequence: SequenceSet) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(sequence.rows.loc[:, ["origin_date", "target_date"]])


def prepare_pre_fit_objects(
    samples: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    contract: LSTMContract,
) -> dict[str, Any]:
    """Fit in-memory scalers and verify all windows; never trains a model."""

    normalized = validate_development_samples(samples, contract=contract)
    registry = load_fold_registry()
    candidate_scalers: dict[tuple[str, str], ScalerRecord] = {}
    refit_scalers: dict[tuple[str, str], ScalerRecord] = {}
    candidate_sequences: dict[tuple[str, str, str, str], SequenceSet] = {}
    refit_sequences: dict[tuple[str, str, str], SequenceSet] = {}
    endpoint_counts: list[dict[str, Any]] = []
    expected_test_parts: list[pd.DataFrame] = []

    for fold in DEVELOPMENT_FOLD_IDS:
        split = prepare_development_fold(normalized, fold, registry=registry)
        for asset in TARGET_ASSETS:
            train = _asset_rows(split.train, asset)
            validation = _asset_rows(split.validation, asset)
            test = _asset_rows(split.test, asset)
            history = _asset_rows(normalized, asset)
            candidate_scaler = fit_scaler_record(
                train,
                fold=fold,
                asset=asset,
                usage="candidate",
                feature_columns=contract.feature_columns,
                data_sha256=PROCESSED_SHA256,
                model_config_sha256=MODEL_SEARCH_SHA256,
            )
            candidate_scalers[(fold, asset)] = candidate_scaler
            refit_rows = pd.concat([train, validation], ignore_index=True)
            refit_scaler = fit_scaler_record(
                refit_rows,
                fold=fold,
                asset=asset,
                usage="refit",
                feature_columns=contract.feature_columns,
                data_sha256=PROCESSED_SHA256,
                model_config_sha256=MODEL_SEARCH_SHA256,
            )
            refit_scalers[(fold, asset)] = refit_scaler

            validation_reference: pd.MultiIndex | None = None
            test_reference: pd.MultiIndex | None = None
            for candidate in FROZEN_CANDIDATES:
                train_sequences = build_sequences(
                    history,
                    train,
                    context_length=candidate.context_length,
                    scaler=candidate_scaler.scaler,
                    require_all_endpoints=False,
                    partition_name="train",
                )
                validation_sequences = build_sequences(
                    history,
                    validation,
                    context_length=candidate.context_length,
                    scaler=candidate_scaler.scaler,
                    require_all_endpoints=True,
                    partition_name="validation",
                )
                test_sequences = build_sequences(
                    history,
                    test,
                    context_length=candidate.context_length,
                    scaler=refit_scaler.scaler,
                    require_all_endpoints=True,
                    partition_name="test",
                )
                candidate_sequences[(fold, asset, candidate.config_id, "train")] = train_sequences
                candidate_sequences[(fold, asset, candidate.config_id, "validation")] = validation_sequences
                candidate_sequences[(fold, asset, candidate.config_id, "test_refit_scaler")] = test_sequences
                combined_sequences = build_sequences(
                    history,
                    refit_rows,
                    context_length=candidate.context_length,
                    scaler=refit_scaler.scaler,
                    require_all_endpoints=False,
                    partition_name="train_plus_validation",
                )
                refit_sequences[(fold, asset, candidate.config_id)] = combined_sequences
                validation_keys = _sequence_keys(validation_sequences)
                test_keys = _sequence_keys(test_sequences)
                if validation_reference is None:
                    validation_reference = validation_keys
                    test_reference = test_keys
                elif not validation_keys.equals(validation_reference) or not test_keys.equals(test_reference):
                    raise Phase2FExecutionError(
                        f"Candidate sequence keys differ for {fold}/{asset}."
                    )
                endpoint_counts.append(
                    {
                        "fold": fold,
                        "asset": asset,
                        "model_config_id": candidate.config_id,
                        "context_length": candidate.context_length,
                        "candidate_scaler_rows": candidate_scaler.row_count,
                        "train_endpoints": len(train_sequences.targets),
                        "train_endpoints_omitted_for_warmup": len(train_sequences.unavailable_keys),
                        "validation_endpoints": len(validation_sequences.targets),
                        "test_endpoints": len(test_sequences.targets),
                        "refit_scaler_rows": refit_scaler.row_count,
                        "refit_endpoints": len(combined_sequences.targets),
                    }
                )
            expected = test.loc[:, ["asset", "origin_date", "target_date", "target_log_return", "target_direction"]].copy()
            expected.insert(0, "fold", fold)
            expected_test_parts.append(expected)

    if len(candidate_scalers) != 20 or len(refit_scalers) != 20:
        raise Phase2FExecutionError("Pre-fit scaler count must be exactly 20 candidate and 20 refit.")
    expected_test = pd.concat(expected_test_parts, ignore_index=True).sort_values(
        list(MATCH_KEYS), kind="mergesort", ignore_index=True
    )
    if len(expected_test) != EXPECTED_BASELINE_ROWS:
        raise Phase2FExecutionError("Expected test endpoints must contain 5,032 rows.")
    baseline_keys = pd.MultiIndex.from_frame(baseline.loc[:, MATCH_KEYS])
    expected_keys = pd.MultiIndex.from_frame(expected_test.loc[:, MATCH_KEYS])
    if not expected_keys.equals(baseline_keys):
        raise Phase2FExecutionError("Constructible test keys do not exactly match the baseline.")
    if not np.allclose(
        expected_test["target_log_return"].to_numpy(float),
        baseline["actual_log_return"].to_numpy(float),
        rtol=1e-12,
        atol=1e-15,
    ) or not np.array_equal(
        expected_test["target_direction"].to_numpy(np.int8),
        baseline["actual_direction"].to_numpy(np.int8),
    ):
        raise Phase2FExecutionError("Constructible test actuals differ from the baseline.")
    return {
        "normalized": normalized,
        "candidate_scalers": candidate_scalers,
        "refit_scalers": refit_scalers,
        "candidate_sequences": candidate_sequences,
        "refit_sequences": refit_sequences,
        "endpoint_counts": endpoint_counts,
        "expected_test": expected_test,
    }


def perform_dry_run(
    verified: Mapping[str, Any],
    environment: Mapping[str, Any],
    *,
    contract: LSTMContract,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = prepare_pre_fit_objects(
        verified["samples"], verified["baseline"], contract=contract
    )
    dry_run = {
        "status": "passed_no_training_performed",
        "run_id": RUN_ID,
        "raw_sha256": RAW_SHA256,
        "processed_sha256": PROCESSED_SHA256,
        "baseline_sha256": BASELINE_SHA256,
        "model_preregistration_sha256": PREREGISTRATION_SHA256,
        "model_search_sha256": MODEL_SEARCH_SHA256,
        "environment_manifest_sha256": environment["manifest_sha256"],
        "torch_version": environment["packages"]["torch_runtime"],
        "execution_device": "cpu",
        "thread_policy": environment["runtime_policy"],
        "feature_count": len(contract.feature_columns),
        "feature_order": list(contract.feature_columns),
        "target_assets": list(contract.assets),
        "market_reference": "SPY_feature_only",
        "candidates": [
            {"config_id": candidate.config_id, "context_length": candidate.context_length}
            for candidate in contract.candidates
        ],
        "seeds": list(contract.seeds),
        "development_folds": list(DEVELOPMENT_FOLD_IDS),
        "candidate_scaler_count": len(prepared["candidate_scalers"]),
        "refit_scaler_count_expected": len(prepared["refit_scalers"]),
        "endpoint_counts": prepared["endpoint_counts"],
        "validation_key_equality_across_candidates_and_seeds": True,
        "test_key_constructibility_for_both_contexts": True,
        "canonical_baseline_rows": len(verified["baseline"]),
        "canonical_baseline_key_match": True,
        "candidate_fits_expected": EXPECTED_CANDIDATE_FITS,
        "winning_refits_expected": EXPECTED_REFITS,
        "canonical_prediction_rows_expected": EXPECTED_PREDICTION_ROWS,
        "final_year_rows_excluded_before_model_validation": verified[
            "final_year_rows_excluded_before_validation"
        ],
        "f1_public_execution_path": False,
        "f1_inaccessible": True,
        "performance_calculated": False,
    }
    return dry_run, prepared


def _assert_environment_only() -> None:
    files = sorted(path.resolve() for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    if files != [ENVIRONMENT_PATH.resolve()]:
        unexpected = [_relative(path) for path in files if path.resolve() != ENVIRONMENT_PATH.resolve()]
        raise Phase2FExecutionError(
            "Refusing to overwrite or mix an existing Phase 2F run; unexpected artifacts: "
            f"{unexpected}."
        )


def _write_scalers(records: Mapping[tuple[str, str], ScalerRecord]) -> dict[tuple[str, str], WrittenArtifact]:
    written: dict[tuple[str, str], WrittenArtifact] = {}
    for key, record in records.items():
        directory = OUTPUT_ROOT / "scalers" / record.usage / record.fold / record.asset
        artifact = write_content_addressed(
            directory,
            f"lstm_{record.usage}_scaler",
            ".json",
            record.payload,
        )
        if artifact.sha256 != record.identifier:
            raise Phase2FExecutionError("Scaler content identity changed during writing.")
        written[key] = artifact
    return written


def _write_checkpoint(
    *,
    stage: str,
    fold: str,
    config_id: str,
    asset: str,
    seed: int,
    payload: bytes,
    expected_sha256: str,
) -> WrittenArtifact:
    directory = OUTPUT_ROOT / "models" / stage / fold / config_id / asset / str(seed)
    artifact = write_content_addressed(directory, "model", ".pt", payload)
    if artifact.sha256 != expected_sha256:
        raise Phase2FExecutionError("Model checkpoint identity changed during writing.")
    return artifact


def run_bounded_reproducibility_check(
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    """Repeat one small fixed slice twice; never repeat the full benchmark."""

    candidate = FROZEN_CANDIDATES[0]
    train_full: SequenceSet = prepared["candidate_sequences"][("D1", "AAPL", candidate.config_id, "train")]
    validation_full: SequenceSet = prepared["candidate_sequences"][("D1", "AAPL", candidate.config_id, "validation")]

    def subset(source: SequenceSet, n: int) -> SequenceSet:
        return SequenceSet(
            features=source.features[:n].copy(),
            targets=source.targets[:n].copy(),
            rows=source.rows.iloc[:n].copy().reset_index(drop=True),
            unavailable_keys=(),
        )

    train = subset(train_full, 96)
    validation = subset(validation_full, 32)
    kwargs = {
        "train": train,
        "validation": validation,
        "run_id": f"{RUN_ID}_bounded_repro",
        "fold": "D1",
        "candidate": candidate,
        "asset": "AAPL",
        "seed": 1729,
        "scaler_identifier": prepared["candidate_scalers"][("D1", "AAPL")].identifier,
        "data_sha256": PROCESSED_SHA256,
        "model_config_sha256": MODEL_SEARCH_SHA256,
    }
    first = fit_candidate(**kwargs)
    second = fit_candidate(**kwargs)
    first_values = first.predictions["predicted_log_return"].to_numpy(float)
    second_values = second.predictions["predicted_log_return"].to_numpy(float)
    predictions_equal = np.array_equal(first_values, second_values)
    epochs_equal = first.audit["best_epoch"] == second.audit["best_epoch"]
    metric_names = ("validation_mae", "validation_rmse", "validation_directional_accuracy")
    metrics_equal = all(first.audit[name] == second.audit[name] for name in metric_names)
    tensors_equal = set(first.state_dict) == set(second.state_dict) and all(
        torch.equal(first.state_dict[name], second.state_dict[name]) for name in first.state_dict
    )
    passed = predictions_equal and epochs_equal and metrics_equal and tensors_equal
    if not passed:
        raise Phase2FExecutionError("Bounded deterministic reproducibility check failed.")
    return {
        "status": "passed",
        "scope": "D1_AAPL_LSTM_21_first_96_train_first_32_validation_rows",
        "seed": 1729,
        "repetitions": 2,
        "prediction_tolerance": 0.0,
        "predictions_bitwise_equal": predictions_equal,
        "best_epoch_equal": epochs_equal,
        "validation_metrics_exactly_equal": metrics_equal,
        "model_state_tensors_exactly_equal": tensors_equal,
        "serialized_pt_byte_equality_required": False,
        "best_epoch": int(first.audit["best_epoch"]),
        "validation_mae": float(first.audit["validation_mae"]),
    }


def independently_verify_saved_results(
    *,
    prediction_artifact: WrittenArtifact,
    metric_artifacts: Mapping[str, WrittenArtifact],
    baseline: pd.DataFrame,
    winners: Mapping[str, LSTMCandidate],
) -> dict[str, Any]:
    """Reload canonical files and independently reconstruct every saved metric level."""

    predictions = pd.read_csv(prediction_artifact.path)
    predictions["origin_date"] = pd.to_datetime(predictions["origin_date"], errors="raise")
    predictions["target_date"] = pd.to_datetime(predictions["target_date"], errors="raise")
    validate_canonical_predictions(predictions, winners)
    assert_exact_baseline_match(predictions, baseline)
    recomputed_per_seed = compute_per_seed_metrics(predictions, baseline, run_id=RUN_ID)
    recomputed_asset = compute_asset_seed_summaries(recomputed_per_seed, run_id=RUN_ID)
    recomputed_macro = compute_macro_by_seed(recomputed_per_seed, run_id=RUN_ID)
    recomputed_macro_summary = compute_macro_seed_summaries(recomputed_macro, run_id=RUN_ID)
    recomputed_development = compute_development_summary(
        recomputed_asset, recomputed_macro_summary, run_id=RUN_ID
    )

    saved_per_seed = pd.read_csv(metric_artifacts["per_seed"].path)
    saved_asset = pd.read_csv(metric_artifacts["asset_seed_summary"].path)
    saved_macro = pd.read_csv(metric_artifacts["macro_by_seed"].path)
    saved_macro_summary = pd.read_csv(metric_artifacts["macro_seed_summary"].path)
    saved_development = pd.read_csv(metric_artifacts["development_summary"].path)
    compare_metric_tables(
        recomputed_per_seed,
        saved_per_seed,
        ["fold", "asset", "seed"],
        list(METRIC_NAMES),
    )
    compare_metric_tables(
        recomputed_asset,
        saved_asset,
        ["fold", "asset"],
        [f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std", "min", "max")],
    )
    compare_metric_tables(
        recomputed_macro,
        saved_macro,
        ["fold", "seed"],
        list(METRIC_NAMES),
    )
    compare_metric_tables(
        recomputed_macro_summary,
        saved_macro_summary,
        ["fold"],
        [f"{metric}_{stat}" for metric in METRIC_NAMES for stat in ("mean", "std", "min", "max")],
    )
    compare_metric_tables(
        recomputed_development,
        saved_development,
        ["scope", "asset", "metric"],
        ["median", "q1", "q3", "iqr"],
    )
    return {
        "status": "passed",
        "prediction_file_reloaded_from_disk": _relative(prediction_artifact.path),
        "prediction_sha256_verified": file_sha256(prediction_artifact.path),
        "prediction_rows": len(predictions),
        "seeds": sorted(int(value) for value in predictions["seed"].unique()),
        "rows_per_seed": {
            str(seed): int((predictions["seed"] == seed).sum()) for seed in FROZEN_SEEDS
        },
        "baseline_key_match_per_seed": True,
        "actual_value_match_per_seed": True,
        "folds": list(DEVELOPMENT_FOLD_IDS),
        "partitions": ["test"],
        "target_years": [2020, 2021, 2022, 2023, 2024],
        "f1_or_2025_rows": 0,
        "model_values": ["lstm"],
        "finite_predictions": True,
        "directions_sign_derived": True,
        "regime_placeholder_only": True,
        "per_seed_metrics_reproduced": True,
        "asset_seed_summaries_reproduced": True,
        "macro_by_seed_reproduced": True,
        "macro_seed_summaries_reproduced": True,
        "development_descriptive_summary_reproduced": True,
        "numeric_tolerance": {"rtol": 1e-12, "atol": 1e-15},
    }


def _artifact_record(artifact: WrittenArtifact) -> dict[str, Any]:
    value: dict[str, Any] = {"path": _relative(artifact.path), "sha256": artifact.sha256}
    if artifact.rows is not None:
        value["rows"] = artifact.rows
    return value


def execute_benchmark(
    *,
    contract: LSTMContract,
    verified: Mapping[str, Any],
    environment: Mapping[str, Any],
    dry_run: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the complete immutable 120-candidate/60-refit Phase 2F run."""

    _assert_environment_only()
    started = perf_counter()
    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    candidate_scaler_artifacts = _write_scalers(prepared["candidate_scalers"])
    if len(candidate_scaler_artifacts) != 20:
        raise Phase2FExecutionError("Candidate scaler write count is not 20.")

    candidate_results = []
    candidate_model_artifacts: list[WrittenArtifact] = []
    completed = 0
    for fold in DEVELOPMENT_FOLD_IDS:
        for candidate in FROZEN_CANDIDATES:
            for asset in TARGET_ASSETS:
                train = prepared["candidate_sequences"][(fold, asset, candidate.config_id, "train")]
                validation = prepared["candidate_sequences"][(fold, asset, candidate.config_id, "validation")]
                scaler = prepared["candidate_scalers"][(fold, asset)]
                for seed in FROZEN_SEEDS:
                    result = fit_candidate_with_retry(
                        train=train,
                        validation=validation,
                        run_id=RUN_ID,
                        fold=fold,
                        candidate=candidate,
                        asset=asset,
                        seed=seed,
                        scaler_identifier=scaler.identifier,
                        data_sha256=PROCESSED_SHA256,
                        model_config_sha256=MODEL_SEARCH_SHA256,
                    )
                    candidate_results.append(result)
                    candidate_model_artifacts.append(
                        _write_checkpoint(
                            stage="candidate",
                            fold=fold,
                            config_id=candidate.config_id,
                            asset=asset,
                            seed=seed,
                            payload=result.checkpoint_payload,
                            expected_sha256=result.checkpoint_identifier,
                        )
                    )
                    completed += 1
                    print(
                        f"Completed candidate fit {completed}/{EXPECTED_CANDIDATE_FITS}: "
                        f"{fold} {candidate.config_id} {asset} seed={seed} "
                        f"best_epoch={result.audit['best_epoch']}",
                        flush=True,
                    )
    candidate_audit, candidate_predictions = candidate_results_frames(candidate_results)
    selection = select_fold_winners(candidate_audit, candidate_predictions, run_id=RUN_ID)
    candidate_prediction_artifact = write_content_addressed(
        OUTPUT_ROOT / "candidates" / "predictions",
        "lstm_validation_predictions",
        ".csv",
        canonical_csv_bytes(candidate_predictions),
        rows=len(candidate_predictions),
    )
    candidate_audit_artifact = write_content_addressed(
        OUTPUT_ROOT / "candidates" / "metrics",
        "lstm_candidate_fit_audit",
        ".csv",
        canonical_csv_bytes(candidate_audit),
        rows=len(candidate_audit),
    )
    selection_artifact = write_content_addressed(
        OUTPUT_ROOT / "selections",
        "lstm_fold_candidates",
        ".csv",
        canonical_csv_bytes(selection.comparison),
        rows=len(selection.comparison),
    )
    best_epoch_artifact = write_content_addressed(
        OUTPUT_ROOT / "selections",
        "lstm_winning_best_epochs",
        ".csv",
        canonical_csv_bytes(selection.best_epochs),
        rows=len(selection.best_epochs),
    )

    refit_scaler_artifacts = _write_scalers(prepared["refit_scalers"])
    if len(refit_scaler_artifacts) != 20:
        raise Phase2FExecutionError("Refit scaler write count is not 20.")
    refit_results = []
    refit_model_artifacts: list[WrittenArtifact] = []
    refit_completed = 0
    for fold in DEVELOPMENT_FOLD_IDS:
        winner = selection.winners[fold]
        for asset in TARGET_ASSETS:
            scaler = prepared["refit_scalers"][(fold, asset)]
            refit_sequences = prepared["refit_sequences"][(fold, asset, winner.config_id)]
            test_sequences = prepared["candidate_sequences"][(fold, asset, winner.config_id, "test_refit_scaler")]
            for seed in FROZEN_SEEDS:
                best = selection.best_epochs.loc[
                    (selection.best_epochs["fold"] == fold)
                    & (selection.best_epochs["asset"] == asset)
                    & (selection.best_epochs["seed"] == seed)
                ]
                if len(best) != 1:
                    raise Phase2FExecutionError("Missing seed-specific winning best epoch.")
                epochs = int(best.iloc[0]["best_epoch"])
                result = refit_and_predict(
                    refit_sequences,
                    test_sequences,
                    run_id=RUN_ID,
                    spec_version=contract.spec_version,
                    data_sha256=PROCESSED_SHA256,
                    model_config_sha256=MODEL_SEARCH_SHA256,
                    fold=fold,
                    candidate=winner,
                    asset=asset,
                    seed=seed,
                    epochs=epochs,
                    scaler_identifier=scaler.identifier,
                )
                refit_results.append(result)
                refit_model_artifacts.append(
                    _write_checkpoint(
                        stage="refit",
                        fold=fold,
                        config_id=winner.config_id,
                        asset=asset,
                        seed=seed,
                        payload=result.checkpoint_payload,
                        expected_sha256=result.model_identifier,
                    )
                )
                refit_completed += 1
                print(
                    f"Completed winner refit {refit_completed}/{EXPECTED_REFITS}: "
                    f"{fold} {winner.config_id} {asset} seed={seed} epochs={epochs}",
                    flush=True,
                )
    refit_audit = pd.DataFrame([result.audit for result in refit_results], columns=REFIT_AUDIT_COLUMNS)
    predictions = pd.concat([result.predictions for result in refit_results], ignore_index=True)
    predictions = predictions.sort_values(
        ["seed", "fold", "asset", "origin_date", "target_date"],
        kind="mergesort",
        ignore_index=True,
    )
    validate_canonical_predictions(predictions, selection.winners)
    assert_exact_baseline_match(predictions, verified["baseline"])
    per_seed_metrics = compute_per_seed_metrics(predictions, verified["baseline"], run_id=RUN_ID)
    asset_seed_summary = compute_asset_seed_summaries(per_seed_metrics, run_id=RUN_ID)
    macro_by_seed = compute_macro_by_seed(per_seed_metrics, run_id=RUN_ID)
    macro_seed_summary = compute_macro_seed_summaries(macro_by_seed, run_id=RUN_ID)
    development_summary = compute_development_summary(
        asset_seed_summary, macro_seed_summary, run_id=RUN_ID
    )

    prediction_artifact = write_content_addressed(
        OUTPUT_ROOT / "predictions",
        "lstm_development_predictions",
        ".csv",
        canonical_csv_bytes(predictions),
        rows=len(predictions),
    )
    refit_audit_artifact = write_content_addressed(
        OUTPUT_ROOT / "refits",
        "lstm_winner_refit_audit",
        ".csv",
        canonical_csv_bytes(refit_audit),
        rows=len(refit_audit),
    )
    tables = {
        "per_seed": ("lstm_metrics_by_asset_seed", per_seed_metrics),
        "asset_seed_summary": ("lstm_metrics_asset_seed_summary", asset_seed_summary),
        "macro_by_seed": ("lstm_metrics_macro_by_seed", macro_by_seed),
        "macro_seed_summary": ("lstm_metrics_macro_seed_summary", macro_seed_summary),
        "development_summary": ("lstm_development_summary", development_summary),
    }
    metric_artifacts: dict[str, WrittenArtifact] = {}
    for key, (stem, frame) in tables.items():
        metric_artifacts[key] = write_content_addressed(
            OUTPUT_ROOT / "metrics",
            stem,
            ".csv",
            canonical_csv_bytes(frame),
            rows=len(frame),
        )

    reproducibility = run_bounded_reproducibility_check(prepared)
    reproducibility_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "lstm_reproducibility_check",
        ".json",
        canonical_json_bytes(reproducibility),
    )
    independent = independently_verify_saved_results(
        prediction_artifact=prediction_artifact,
        metric_artifacts=metric_artifacts,
        baseline=verified["baseline"],
        winners=selection.winners,
    )
    independent_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "lstm_independent_verification",
        ".json",
        canonical_json_bytes(independent),
    )
    final_lightgbm_tree = _lightgbm_tree_identity()
    if final_lightgbm_tree != verified["lightgbm_tree"]:
        raise Phase2FExecutionError("Frozen LightGBM artifacts changed during Phase 2F.")

    runtime_seconds = perf_counter() - started
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_timestamp_utc": created_at,
        "experiment_spec_version": contract.spec_version,
        "status": "completed",
        "source_tree": _source_tree_identity(),
        "frozen_inputs": {
            "raw_dataset": {"path": _relative(RAW_DATA_PATH), "sha256": RAW_SHA256},
            "processed_dataset": {"path": _relative(PROCESSED_DATA_PATH), "sha256": PROCESSED_SHA256},
            "baseline": {"path": _relative(BASELINE_PATH), "sha256": BASELINE_SHA256},
            "model_preregistration": {"path": _relative(PREREGISTRATION_PATH), "sha256": PREREGISTRATION_SHA256},
            "model_search": {"path": _relative(MODEL_SEARCH_PATH), "sha256": MODEL_SEARCH_SHA256},
            "lightgbm_predictions_provenance_only": {"path": _relative(LIGHTGBM_PREDICTION_PATH), "sha256": LIGHTGBM_PREDICTION_SHA256},
            "lightgbm_run_manifest_provenance_only": {"path": _relative(LIGHTGBM_MANIFEST_PATH), "sha256": LIGHTGBM_MANIFEST_SHA256},
            "lightgbm_tree": final_lightgbm_tree,
        },
        "environment": {
            "manifest": {"path": environment["manifest_path"], "sha256": environment["manifest_sha256"]},
            "packages": environment["packages"],
            "python_version": environment["python_version"],
            "python_executable": environment["python_executable"],
            "device": "cpu",
            "thread_policy": environment["runtime_policy"],
            "strict_determinism_achieved": True,
        },
        "contract": {
            "feature_order": list(contract.feature_columns),
            "target_assets": list(contract.assets),
            "market_reference": "SPY_feature_only",
            "candidates": [candidate.__dict__ for candidate in contract.candidates],
            "seeds": list(contract.seeds),
            "architecture": {
                "input_shape": "batch_context_17",
                "lstm_layers": 1,
                "hidden_size": 32,
                "bidirectional": False,
                "internal_dropout": 0.0,
                "post_lstm_dropout": 0.20,
                "output": "Linear(32,1)",
                "dtype": "float32",
            },
            "optimization": {
                "loss": "L1Loss",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "batch_size": 64,
                "max_epochs": 60,
                "gradient_clip_norm": 1.0,
                "shuffle": False,
                "drop_last": False,
                "num_workers": 0,
                "target_scaling": "none",
            },
            "early_stopping": {
                "monitor": "row_level_validation_mae",
                "min_delta": 0.00001,
                "patience": 8,
                "best_epoch_numbering": "one_based",
                "restore_best_state": True,
            },
            "selection": {
                "primary": "seed_mean_within_asset_then_equal_asset_macro_validation_mae",
                "tie_threshold": 0.00001,
                "tie_break": ["shorter_context", "macro_validation_rmse", "lexical_config_id"],
            },
            "refit": "fresh_per_seed_train_plus_validation_for_seed_specific_best_epoch",
            "prediction_schema": list(CANONICAL_PREDICTION_FIELDS),
            "regime_placeholder": REGIME_PLACEHOLDER,
        },
        "folds": list(DEVELOPMENT_FOLD_IDS),
        "fold_endpoint_counts": dry_run["endpoint_counts"],
        "selected_context_by_fold": {
            fold: selection.winners[fold].config_id for fold in DEVELOPMENT_FOLD_IDS
        },
        "candidate_fit_statuses": candidate_audit.to_dict(orient="records"),
        "candidate_best_epochs": selection.best_epochs.to_dict(orient="records"),
        "refit_statuses": refit_audit.to_dict(orient="records"),
        "counts": {
            "candidate_scalers": len(candidate_scaler_artifacts),
            "candidate_fits": len(candidate_audit),
            "candidate_validation_prediction_rows": len(candidate_predictions),
            "refit_scalers": len(refit_scaler_artifacts),
            "winning_refits": len(refit_audit),
            "canonical_prediction_rows": len(predictions),
            "rows_per_seed": {
                str(seed): int((predictions["seed"] == seed).sum()) for seed in FROZEN_SEEDS
            },
        },
        "artifacts": {
            "candidate_scalers": [_artifact_record(value) for value in candidate_scaler_artifacts.values()],
            "refit_scalers": [_artifact_record(value) for value in refit_scaler_artifacts.values()],
            "candidate_models": [_artifact_record(value) for value in candidate_model_artifacts],
            "refit_models": [_artifact_record(value) for value in refit_model_artifacts],
            "candidate_predictions": _artifact_record(candidate_prediction_artifact),
            "candidate_fit_audit": _artifact_record(candidate_audit_artifact),
            "selection": _artifact_record(selection_artifact),
            "winning_best_epochs": _artifact_record(best_epoch_artifact),
            "refit_audit": _artifact_record(refit_audit_artifact),
            "predictions": _artifact_record(prediction_artifact),
            "metrics": {name: _artifact_record(value) for name, value in metric_artifacts.items()},
            "reproducibility": _artifact_record(reproducibility_artifact),
            "independent_verification": _artifact_record(independent_artifact),
        },
        "verification": {
            "reproducibility": reproducibility,
            "independent_saved_file_verification": independent,
            "canonical_baseline_key_match_every_seed": True,
            "lightgbm_artifacts_unchanged": True,
        },
        "runtime": {
            "seconds": runtime_seconds,
            "candidate_parallelism": "sequential",
            "refit_parallelism": "sequential",
            "warnings": [],
            "errors": [],
        },
        "protected_boundaries": {
            "f1_status": "LOCKED_UNTOUCHED",
            "f1_or_2025_performance_evaluated": False,
            "patchtst_status": "CLOSED_NOT_IMPLEMENTED",
            "regime_analysis_performed": False,
            "regime_fields": "sentinel_only",
        },
        "specification_deviations": [],
    }
    manifest_payload = canonical_json_bytes(manifest)
    manifest_artifact = write_content_addressed(
        OUTPUT_ROOT, "run_manifest", ".json", manifest_payload
    )
    return {
        "manifest": manifest,
        "manifest_artifact": manifest_artifact,
        "candidate_audit": candidate_audit,
        "candidate_predictions": candidate_predictions,
        "selection": selection,
        "best_epochs": selection.best_epochs,
        "refit_audit": refit_audit,
        "predictions": predictions,
        "per_seed_metrics": per_seed_metrics,
        "asset_seed_summary": asset_seed_summary,
        "macro_by_seed": macro_by_seed,
        "macro_seed_summary": macro_seed_summary,
        "development_summary": development_summary,
        "artifacts": {
            "predictions": prediction_artifact,
            "candidate_predictions": candidate_prediction_artifact,
            "candidate_audit": candidate_audit_artifact,
            "selection": selection_artifact,
            "best_epochs": best_epoch_artifact,
            "refit_audit": refit_audit_artifact,
            "metrics": metric_artifacts,
            "reproducibility": reproducibility_artifact,
            "independent": independent_artifact,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Phase 2F D1-D5 LSTM benchmark.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify environment, scalers, sequence coverage, and counts without training.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    environment = verify_exact_environment()
    contract = load_lstm_contract(MODEL_SEARCH_PATH)
    verified = verify_frozen_inputs(contract)
    dry_run, prepared = perform_dry_run(
        verified, environment, contract=contract
    )
    if args.dry_run:
        print(json.dumps(dry_run, indent=2, default=str), flush=True)
        return 0
    result = execute_benchmark(
        contract=contract,
        verified=verified,
        environment=environment,
        dry_run=dry_run,
        prepared=prepared,
    )
    counts = result["manifest"]["counts"]
    print(f"Phase 2F run manifest: {_relative(result['manifest_artifact'].path)}")
    print(f"Canonical prediction artifact: {_relative(result['artifacts']['predictions'].path)}")
    print(f"Candidate fits completed: {counts['candidate_fits']}/120")
    print(f"Winning refits completed: {counts['winning_refits']}/60")
    print(f"Canonical prediction rows: {counts['canonical_prediction_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
