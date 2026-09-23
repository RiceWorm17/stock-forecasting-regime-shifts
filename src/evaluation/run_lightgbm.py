"""One-command execution of the frozen Phase 2E LightGBM benchmark.

Run the no-fit gate first with::

    python -m src.evaluation.run_lightgbm --dry-run

Run the preregistered D1--D5 benchmark with::

    python -m src.evaluation.run_lightgbm

There is deliberately no fold selector and no F1/2025 execution path.
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
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.integrity import file_sha256, verify_snapshot
from src.evaluation.lightgbm_evaluation import (
    BEST_ITERATION_COLUMNS,
    CANDIDATE_METRIC_COLUMNS,
    MACRO_METRIC_COLUMNS,
    PER_ASSET_METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    REFIT_AUDIT_COLUMNS,
    REGIME_PLACEHOLDER,
    SUMMARY_COLUMNS,
    LightGBMArtifacts,
    LightGBMDevelopmentRun,
    canonical_json_bytes,
    prepare_development_fold,
    run_development_lightgbm,
    sha256_bytes,
    validate_baseline_predictions,
    validate_development_samples,
    write_content_addressed_bytes,
    write_development_artifacts,
)
from src.evaluation.walk_forward import DEVELOPMENT_FOLD_IDS, load_fold_registry
from src.models.lightgbm_model import (
    FROZEN_FEATURE_COLUMNS,
    LightGBMContract,
    build_lgbm_regressor,
    load_lightgbm_contract,
    load_lightgbm_runtime,
    resolve_estimator_parameters,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "lightgbm"

RAW_SHA256 = "55f74f058493102387f1c2d0a848bd673a3cfa0473d617dfdb34aea8a94f7736"
PROCESSED_SHA256 = "9352da6b434011f9844ef0acc1813ea33dce39152712752bba21a3005aabf7be"
BASELINE_SHA256 = "e09e4909e9af7b7836c5e0b2c8d7b320d2848d5da9b33b8c9492421e0a108bb4"
PREREGISTRATION_SHA256 = (
    "be04b9ce8457c5b26a8042f548eccce8ddb4b5d1b86a1404a43ef1bb1a52d613"
)
MODEL_SEARCH_SHA256 = "e15e42ff2b2f4f2bc7ccc185195c26fd35851b259bc28e08a6ba2149b6aaf38c"
PYPROJECT_SHA256 = "c7463defacea5d7767d8fb693beedaae558005f5e1d2c27cae8f07a5cfe01fbf"
ENVIRONMENT_SHA256 = "db01e5e3baae9e6e582480ec09cf2a137c452819a65992b71f0c4654eec3beac"
EXACT_LIGHTGBM_VERSION = "4.7.0"
EXPECTED_PREDICTION_ROWS = 5_032
RUN_ID = f"phase2e_lgbm_dev_{PROCESSED_SHA256[:12]}"

RAW_DATA_PATH = REPOSITORY_ROOT / "data" / "raw" / (
    "yfinance_daily_2015-01-01_2025-12-31_20260913T152313Z_55f74f058493.csv"
)
RAW_MANIFEST_PATH = RAW_DATA_PATH.with_suffix(".manifest.json")
PROCESSED_DATA_PATH = REPOSITORY_ROOT / "data" / "processed" / (
    f"supervised_v1_{PROCESSED_SHA256}.csv"
)
PROCESSED_MANIFEST_PATH = PROCESSED_DATA_PATH.with_suffix(".manifest.json")
BASELINE_PATH = REPOSITORY_ROOT / "results" / "baselines" / "predictions" / (
    f"baseline_predictions_{BASELINE_SHA256}.csv"
)
PREREGISTRATION_PATH = REPOSITORY_ROOT / "docs" / "MODEL_PREREGISTRATION.md"
MODEL_SEARCH_PATH = REPOSITORY_ROOT / "configs" / "model_search.yaml"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
ENVIRONMENT_PATH = OUTPUT_ROOT / "environment" / (
    f"lightgbm_environment_{ENVIRONMENT_SHA256}.json"
)

FLOAT_RTOL = 1e-12
FLOAT_ATOL = 1e-15


class Phase2EExecutionError(RuntimeError):
    """Raised when a frozen Phase 2E gate or post-run verification fails."""


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Phase2EExecutionError(f"Unable to read JSON object {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Phase2EExecutionError(f"JSON root in {path} must be an object.")
    return payload


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise Phase2EExecutionError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}."
        )
    return actual


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise Phase2EExecutionError(f"Required distribution {name!r} is unavailable.") from exc


def verify_exact_environment() -> dict[str, Any]:
    """Verify the already-frozen runtime before any LightGBM fit is allowed."""

    environment_hash = _require_hash(
        ENVIRONMENT_PATH, ENVIRONMENT_SHA256, "LightGBM environment manifest"
    )
    manifest = _load_json_object(ENVIRONMENT_PATH)
    runtime = load_lightgbm_runtime()
    if runtime.version != EXACT_LIGHTGBM_VERSION:
        raise Phase2EExecutionError(
            f"LightGBM must be exactly {EXACT_LIGHTGBM_VERSION}; imported {runtime.version}."
        )

    recorded_packages = manifest.get("packages")
    if not isinstance(recorded_packages, Mapping):
        raise Phase2EExecutionError("Environment manifest has no package map.")
    observed_packages = {
        "lightgbm": runtime.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": _distribution_version("PyYAML"),
        "scikit_learn": _distribution_version("scikit-learn"),
        "scipy": _distribution_version("scipy"),
    }
    if dict(recorded_packages) != observed_packages:
        raise Phase2EExecutionError(
            "Imported package versions differ from the frozen environment: "
            f"recorded={dict(recorded_packages)}, observed={observed_packages}."
        )

    recorded_python = manifest.get("python")
    if not isinstance(recorded_python, Mapping):
        raise Phase2EExecutionError("Environment manifest has no Python runtime record.")
    observed_python = platform.python_version()
    if observed_python != recorded_python.get("version"):
        raise Phase2EExecutionError(
            f"Python version differs from the environment lock: {observed_python}."
        )
    if Path(sys.executable).resolve() != Path(str(recorded_python.get("executable"))).resolve():
        raise Phase2EExecutionError("Python executable differs from the frozen environment.")
    if platform.python_compiler() != recorded_python.get("compiler"):
        raise Phase2EExecutionError("Python compiler differs from the frozen environment.")

    recorded_platform = manifest.get("platform")
    if not isinstance(recorded_platform, Mapping):
        raise Phase2EExecutionError("Environment manifest has no platform record.")
    platform_checks = {
        "operating_system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform_string": platform.platform(),
    }
    for key, observed in platform_checks.items():
        if recorded_platform.get(key) != observed:
            raise Phase2EExecutionError(
                f"Platform field {key} differs from the frozen environment: {observed!r}."
            )

    installation = manifest.get("installation")
    if not isinstance(installation, Mapping):
        raise Phase2EExecutionError("Environment manifest has no installation record.")
    dependency_target = Path(str(installation.get("dependency_target"))).resolve()
    module_path = Path(str(runtime.module.__file__)).resolve()
    if not dependency_target.is_dir() or not module_path.is_relative_to(dependency_target):
        raise Phase2EExecutionError(
            "LightGBM was not imported from the already-frozen isolated dependency target: "
            f"{module_path}."
        )

    thread_policy = manifest.get("thread_policy")
    if not isinstance(thread_policy, Mapping) or dict(thread_policy) != {
        "candidate_parallelism": "sequential",
        "lightgbm_n_jobs": 1,
        "refit_parallelism": "sequential",
    }:
        raise Phase2EExecutionError("Frozen LightGBM thread policy has drifted.")
    if manifest.get("device_policy") != "cpu_only_reference_run":
        raise Phase2EExecutionError("Frozen LightGBM run must use the CPU-only policy.")

    repository_inputs = manifest.get("repository_inputs")
    if not isinstance(repository_inputs, Mapping):
        raise Phase2EExecutionError("Environment manifest lacks repository input identities.")
    expected_repository_inputs = {
        "model_preregistration_path": "docs/MODEL_PREREGISTRATION.md",
        "model_preregistration_sha256": PREREGISTRATION_SHA256,
        "model_search_path": "configs/model_search.yaml",
        "model_search_sha256": MODEL_SEARCH_SHA256,
        "pyproject_path": "pyproject.toml",
        "pyproject_sha256": PYPROJECT_SHA256,
    }
    if dict(repository_inputs) != expected_repository_inputs:
        raise Phase2EExecutionError("Environment manifest repository identities have drifted.")

    return {
        "manifest": manifest,
        "manifest_path": _relative(ENVIRONMENT_PATH),
        "manifest_sha256": environment_hash,
        "packages": observed_packages,
        "python_version": observed_python,
        "python_executable": str(Path(sys.executable).resolve()),
        "lightgbm_module_path": str(module_path),
        "dependency_target": str(dependency_target),
        "device_policy": "cpu_only_reference_run",
        "n_jobs": 1,
    }


def verify_frozen_inputs(contract: LightGBMContract) -> dict[str, Any]:
    """Verify pinned files and load only D1--D5 model-development rows."""

    _require_hash(PREREGISTRATION_PATH, PREREGISTRATION_SHA256, "Model preregistration")
    _require_hash(MODEL_SEARCH_PATH, MODEL_SEARCH_SHA256, "Model-search configuration")
    _require_hash(PYPROJECT_PATH, PYPROJECT_SHA256, "Phase 2E pyproject")
    if contract.raw_sha256 != RAW_SHA256 or contract.processed_sha256 != PROCESSED_SHA256:
        raise Phase2EExecutionError("Loaded LightGBM contract has unexpected data identities.")

    raw_verification = verify_snapshot(RAW_DATA_PATH, RAW_MANIFEST_PATH)
    if raw_verification.sha256 != RAW_SHA256:
        raise Phase2EExecutionError("Raw snapshot differs from the preregistered hash.")

    processed_manifest = _load_json_object(PROCESSED_MANIFEST_PATH)
    processed_hash = _require_hash(
        PROCESSED_DATA_PATH, PROCESSED_SHA256, "Processed supervised dataset"
    )
    if processed_manifest.get("file_sha256") != PROCESSED_SHA256:
        raise Phase2EExecutionError("Processed manifest contains an unexpected data hash.")
    if processed_manifest.get("processed_file") != PROCESSED_DATA_PATH.name:
        raise Phase2EExecutionError("Processed manifest names a different data file.")
    source = processed_manifest.get("source")
    if not isinstance(source, Mapping) or dict(source) != {
        "raw_file": RAW_DATA_PATH.name,
        "raw_file_sha256": RAW_SHA256,
    }:
        raise Phase2EExecutionError("Processed-to-raw provenance link is invalid.")
    if processed_manifest.get("experiment_spec_version") != contract.spec_version:
        raise Phase2EExecutionError("Processed artifact has the wrong specification version.")

    _require_hash(BASELINE_PATH, BASELINE_SHA256, "Canonical Phase 2C baseline")
    baseline = validate_baseline_predictions(
        pd.read_csv(BASELINE_PATH), data_version=PROCESSED_SHA256
    )
    if len(baseline) != EXPECTED_PREDICTION_ROWS:
        raise Phase2EExecutionError(
            f"Canonical baseline must contain {EXPECTED_PREDICTION_ROWS} rows."
        )

    all_samples = pd.read_csv(PROCESSED_DATA_PATH)
    target_dates = pd.to_datetime(all_samples["target_date"], errors="raise")
    development_mask = target_dates.dt.year <= 2024
    if not development_mask.any() or not (~development_mask).any():
        raise Phase2EExecutionError(
            "Frozen processed data must contain both development rows and mechanically "
            "retained final-year rows."
        )
    development_samples = all_samples.loc[development_mask].copy()
    development_samples["target_date"] = target_dates.loc[development_mask]

    return {
        "contract": contract,
        "samples": development_samples,
        "baseline": baseline,
        "raw_verification": raw_verification.to_dict(),
        "raw_path": RAW_DATA_PATH,
        "raw_manifest_path": RAW_MANIFEST_PATH,
        "processed_path": PROCESSED_DATA_PATH,
        "processed_manifest_path": PROCESSED_MANIFEST_PATH,
        "processed_manifest": processed_manifest,
        "processed_sha256": processed_hash,
        "baseline_path": BASELINE_PATH,
        "baseline_sha256": BASELINE_SHA256,
        "final_year_rows_excluded_before_model_validation": int((~development_mask).sum()),
    }


def _keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame.loc[:, ["fold", "asset", "origin_date", "target_date"]]
    )


def _fold_partition_counts(
    samples: pd.DataFrame, contract: LightGBMContract
) -> tuple[dict[str, Any], pd.DataFrame]:
    registry = load_fold_registry()
    counts: dict[str, Any] = {}
    expected_test_rows: list[pd.DataFrame] = []
    for fold_id in DEVELOPMENT_FOLD_IDS:
        split = prepare_development_fold(
            samples, fold_id, contract=contract, registry=registry
        )
        counts[fold_id] = {}
        for partition_name in ("train", "validation", "test"):
            frame = getattr(split, partition_name)
            counts[fold_id][partition_name] = {
                "total": int(len(frame)),
                "by_asset": {
                    asset: int((frame["asset"] == asset).sum())
                    for asset in contract.assets
                },
                "target_date_start": frame["target_date"].min().date().isoformat(),
                "target_date_end": frame["target_date"].max().date().isoformat(),
            }
        expected = split.test.loc[
            :, ["asset", "origin_date", "target_date", "target_log_return", "target_direction"]
        ].copy()
        expected.insert(0, "fold", fold_id)
        expected_test_rows.append(expected)
    return counts, pd.concat(expected_test_rows, ignore_index=True)


def perform_dry_run(
    verified_inputs: Mapping[str, Any], environment: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate all pre-fit gates without constructing or fitting an estimator."""

    contract: LightGBMContract = verified_inputs["contract"]
    normalized = validate_development_samples(
        verified_inputs["samples"], contract=contract
    )
    counts, expected_test = _fold_partition_counts(normalized, contract)
    baseline: pd.DataFrame = verified_inputs["baseline"]
    expected_test = expected_test.sort_values(
        ["fold", "asset", "origin_date", "target_date"],
        kind="mergesort",
        ignore_index=True,
    )
    baseline_keys = _keys(baseline)
    expected_keys = _keys(expected_test)
    if len(expected_test) != EXPECTED_PREDICTION_ROWS or set(expected_keys) != set(
        baseline_keys
    ):
        raise Phase2EExecutionError("Dry-run test keys do not exactly match Phase 2C.")
    baseline_actual = baseline.merge(
        expected_test,
        on=["fold", "asset", "origin_date", "target_date"],
        how="left",
        validate="one_to_one",
        suffixes=("_baseline", "_source"),
    )
    if not np.allclose(
        baseline_actual["actual_log_return"].to_numpy(dtype="float64"),
        baseline_actual["target_log_return"].to_numpy(dtype="float64"),
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
    ) or not np.array_equal(
        baseline_actual["actual_direction"].to_numpy(dtype="int8"),
        baseline_actual["target_direction"].to_numpy(dtype="int8"),
    ):
        raise Phase2EExecutionError("Dry-run baseline actual values differ from source rows.")

    f1_inaccessible = False
    try:
        prepare_development_fold(normalized, "F1", contract=contract)
    except ValueError:
        f1_inaccessible = True
    if not f1_inaccessible:
        raise Phase2EExecutionError("F1 unexpectedly became available to Phase 2E.")

    required_audit_fields = {
        "run_id",
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
        "actual_positive_rate",
        "fit_runtime_seconds",
        "status",
        "warning_status",
        "warning_messages",
        "error_status",
        "error_type",
        "error_message",
    }
    if not required_audit_fields.issubset(CANDIDATE_METRIC_COLUMNS):
        raise Phase2EExecutionError("Candidate audit schema is incomplete before fitting.")

    return {
        "status": "passed_no_training_performed",
        "run_id": RUN_ID,
        "raw_sha256": RAW_SHA256,
        "processed_sha256": PROCESSED_SHA256,
        "model_preregistration_sha256": PREREGISTRATION_SHA256,
        "model_search_sha256": MODEL_SEARCH_SHA256,
        "baseline_path": _relative(BASELINE_PATH),
        "baseline_sha256": BASELINE_SHA256,
        "baseline_rows": int(len(baseline)),
        "baseline_key_match": True,
        "lightgbm_version": environment["packages"]["lightgbm"],
        "feature_count": len(contract.feature_columns),
        "feature_order": list(contract.feature_columns),
        "eligible_development_rows": int(normalized["core_evaluation_eligible"].sum()),
        "loaded_development_rows": int(len(normalized)),
        "final_year_rows_excluded_before_model_validation": verified_inputs[
            "final_year_rows_excluded_before_model_validation"
        ],
        "fold_partition_counts": counts,
        "candidate_count": len(contract.candidates),
        "candidate_fit_count_expected": len(contract.candidates)
        * len(DEVELOPMENT_FOLD_IDS)
        * len(contract.assets),
        "winning_refit_count_expected": len(DEVELOPMENT_FOLD_IDS)
        * len(contract.assets),
        "candidate_audit_fields_complete": True,
        "development_folds": list(DEVELOPMENT_FOLD_IDS),
        "f1_public_execution_path": False,
        "f1_inaccessible": True,
        "performance_calculated": False,
    }


def _assert_only_environment_exists() -> None:
    existing = sorted(path.resolve() for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    allowed = [ENVIRONMENT_PATH.resolve()]
    if existing != allowed:
        unexpected = [_relative(path) for path in existing if path not in allowed]
        raise Phase2EExecutionError(
            "Refusing to start another Phase 2E benchmark because generated LightGBM "
            f"artifacts already exist: {unexpected}."
        )


def _progress(event: Mapping[str, Any]) -> None:
    label = "candidate fit" if event["stage"] == "candidate" else "winner refit"
    print(
        f"Completed {label} {event['completed']}/{event['total']}: "
        f"{event['fold']} {event['model_config_id']} {event['asset']}",
        flush=True,
    )


def _source_tree_identity() -> dict[str, Any]:
    files = sorted(
        (path for path in (REPOSITORY_ROOT / "src").rglob("*.py") if path.is_file()),
        key=lambda item: item.relative_to(REPOSITORY_ROOT).as_posix(),
    )
    per_file: dict[str, str] = {}
    aggregate = hashlib.sha256()
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
        "file_count": len(per_file),
        "files": per_file,
        "git_revision": None,
        "git_repository_available": False,
    }


def run_bounded_reproducibility_check(
    contract: LightGBMContract,
) -> dict[str, Any]:
    """Fit one small synthetic configuration twice after the real benchmark."""

    generator = np.random.default_rng(1729)
    features = pd.DataFrame(
        generator.normal(size=(160, len(contract.feature_columns))),
        columns=list(contract.feature_columns),
        dtype="float64",
    )
    coefficients = np.linspace(-0.003, 0.003, len(contract.feature_columns))
    target = features.to_numpy() @ coefficients + generator.normal(0.0, 0.001, 160)
    train_x = features.iloc[:128]
    train_y = target[:128]
    check_x = features.iloc[128:]
    candidate = contract.candidate("LGBM_03")
    predictions: list[np.ndarray] = []
    model_digests: list[str] = []
    for _ in range(2):
        estimator = build_lgbm_regressor(contract, candidate, n_estimators=25)
        estimator.fit(train_x, train_y)
        prediction = np.asarray(estimator.predict(check_x), dtype="float64")
        predictions.append(prediction)
        model_text = estimator.booster_.model_to_string(num_iteration=25)
        model_digests.append(sha256_bytes(model_text.encode("utf-8")))
    difference = np.abs(predictions[0] - predictions[1])
    maximum = float(difference.max(initial=0.0))
    exact_equal = bool(np.array_equal(predictions[0], predictions[1]))
    tolerance = 1e-15
    within_tolerance = bool(maximum <= tolerance)
    if not within_tolerance:
        raise Phase2EExecutionError(
            f"Bounded deterministic check failed; maximum difference={maximum}."
        )
    return {
        "status": "passed",
        "scope": "one_bounded_synthetic_equivalent_fit_repeated_twice",
        "full_benchmark_rerun": False,
        "additional_fit_count": 2,
        "config_id": candidate.config_id,
        "n_train": len(train_x),
        "n_check": len(check_x),
        "n_estimators": 25,
        "seed": 1729,
        "n_jobs": 1,
        "feature_order": list(contract.feature_columns),
        "exact_prediction_equality": exact_equal,
        "maximum_absolute_prediction_difference": maximum,
        "absolute_tolerance": tolerance,
        "within_tolerance": within_tolerance,
        "model_text_sha256": model_digests,
        "model_text_identical": model_digests[0] == model_digests[1],
    }


def _assert_close(
    observed: np.ndarray,
    expected: np.ndarray,
    label: str,
    *,
    rtol: float = FLOAT_RTOL,
    atol: float = FLOAT_ATOL,
) -> float:
    if observed.shape != expected.shape:
        raise Phase2EExecutionError(
            f"{label} shape mismatch: {observed.shape} != {expected.shape}."
        )
    if not np.allclose(observed, expected, rtol=rtol, atol=atol, equal_nan=True):
        maximum = float(np.nanmax(np.abs(observed - expected)))
        raise Phase2EExecutionError(
            f"{label} failed strict recomputation; maximum absolute difference={maximum}."
        )
    finite_difference = np.abs(observed - expected)
    finite_difference = finite_difference[np.isfinite(finite_difference)]
    return float(finite_difference.max(initial=0.0))


def _read_saved_table(artifact: Any) -> pd.DataFrame:
    if file_sha256(artifact.path) != artifact.sha256:
        raise Phase2EExecutionError(f"Saved artifact hash failed: {artifact.path}.")
    return pd.read_csv(artifact.path)


def independently_verify_saved_results(
    verified_inputs: Mapping[str, Any],
    artifacts: LightGBMArtifacts,
) -> dict[str, Any]:
    """Reload saved rows and recompute every headline metric independently."""

    tables = {name: _read_saved_table(item) for name, item in artifacts.tables.items()}
    predictions = tables["predictions"]
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise Phase2EExecutionError("Saved learned predictions have the wrong 18 columns.")
    for column in ("origin_date", "target_date"):
        predictions[column] = pd.to_datetime(predictions[column], errors="raise")
    if len(predictions) != EXPECTED_PREDICTION_ROWS:
        raise Phase2EExecutionError("Saved learned predictions do not contain 5,032 rows.")
    if predictions.duplicated(["fold", "asset", "origin_date", "target_date"]).any():
        raise Phase2EExecutionError("Saved learned predictions contain duplicate keys.")
    if set(predictions["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise Phase2EExecutionError("Saved learned predictions are not exactly D1--D5.")
    if not predictions["partition"].eq("test").all():
        raise Phase2EExecutionError("Saved learned predictions contain non-test rows.")
    if set(predictions["target_date"].dt.year) != {2020, 2021, 2022, 2023, 2024}:
        raise Phase2EExecutionError("Saved learned predictions have unexpected target years.")
    if not predictions["seed"].eq(1729).all():
        raise Phase2EExecutionError("Saved learned predictions do not use only seed 1729.")
    if not predictions["model"].eq("lightgbm").all():
        raise Phase2EExecutionError("Saved learned predictions have the wrong model ID.")
    if not predictions[
        ["trend_regime", "volatility_regime", "transition_regime"]
    ].eq(REGIME_PLACEHOLDER).all().all():
        raise Phase2EExecutionError("Saved regime fields do not contain only the sentinel.")
    numeric_predictions = predictions[
        ["actual_log_return", "predicted_log_return"]
    ].to_numpy(dtype="float64")
    if not np.isfinite(numeric_predictions).all():
        raise Phase2EExecutionError("Saved learned return values contain non-finite values.")
    expected_direction = (
        predictions["predicted_log_return"].to_numpy(dtype="float64") > 0.0
    ).astype("int8")
    if not np.array_equal(
        predictions["predicted_direction"].to_numpy(dtype="int8"), expected_direction
    ):
        raise Phase2EExecutionError("Saved predicted directions are not sign-derived.")

    selections = tables["selections"]
    selected_mask = selections["selected"].astype(str).str.lower().eq("true")
    winners = selections.loc[selected_mask, ["fold", "model_config_id"]]
    if len(winners) != 5 or winners["fold"].nunique() != 5:
        raise Phase2EExecutionError("Saved selection table lacks one winner per fold.")
    winner_map = dict(zip(winners["fold"], winners["model_config_id"], strict=True))
    for fold_id, config_id in winner_map.items():
        if not predictions.loc[
            predictions["fold"] == fold_id, "model_config_id"
        ].eq(config_id).all():
            raise Phase2EExecutionError(
                f"Saved prediction config disagrees with the {fold_id} winner."
            )

    baseline = verified_inputs["baseline"].copy()
    baseline_keys = _keys(baseline)
    learned_keys = _keys(predictions)
    if len(baseline) != len(predictions) or set(baseline_keys) != set(learned_keys):
        raise Phase2EExecutionError("Saved learned/baseline keys do not match exactly.")
    baseline_columns = [
        "fold",
        "asset",
        "origin_date",
        "target_date",
        "actual_log_return",
        "predicted_log_return",
        "actual_direction",
        "predicted_direction",
    ]
    matched = predictions.merge(
        baseline.loc[:, baseline_columns],
        on=["fold", "asset", "origin_date", "target_date"],
        how="left",
        validate="one_to_one",
        suffixes=("_model", "_baseline"),
    )
    if not np.allclose(
        matched["actual_log_return_model"].to_numpy(dtype="float64"),
        matched["actual_log_return_baseline"].to_numpy(dtype="float64"),
        rtol=FLOAT_RTOL,
        atol=FLOAT_ATOL,
    ) or not np.array_equal(
        matched["actual_direction_model"].to_numpy(dtype="int8"),
        matched["actual_direction_baseline"].to_numpy(dtype="int8"),
    ):
        raise Phase2EExecutionError("Learned and baseline actual values differ.")

    per_asset_rows: list[dict[str, Any]] = []
    for (fold_id, asset), frame in matched.groupby(["fold", "asset"], sort=True):
        actual = frame["actual_log_return_model"].to_numpy(dtype="float64")
        predicted = frame["predicted_log_return_model"].to_numpy(dtype="float64")
        baseline_predicted = frame["predicted_log_return_baseline"].to_numpy(
            dtype="float64"
        )
        actual_direction = frame["actual_direction_model"].to_numpy(dtype="int8")
        predicted_direction = frame["predicted_direction_model"].to_numpy(dtype="int8")
        baseline_direction = frame["predicted_direction_baseline"].to_numpy(dtype="int8")
        absolute_errors = np.abs(predicted - actual)
        baseline_absolute_errors = np.abs(baseline_predicted - actual)
        mae = float(absolute_errors.mean())
        baseline_mae = float(baseline_absolute_errors.mean())
        directional_accuracy = float(np.mean(predicted_direction == actual_direction))
        baseline_da = float(np.mean(baseline_direction == actual_direction))
        positive_count = int(actual_direction.sum())
        first = frame.iloc[0]
        per_asset_rows.append(
            {
                "run_id": first["run_id"],
                "spec_version": first["spec_version"],
                "data_version": first["data_version"],
                "model": "lightgbm",
                "model_config_id": first["model_config_id"],
                "seed": 1729,
                "fold": fold_id,
                "asset": asset,
                "n": len(frame),
                "mae": mae,
                "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
                "directional_accuracy": directional_accuracy,
                "actual_positive_count": positive_count,
                "actual_nonpositive_count": len(frame) - positive_count,
                "actual_positive_rate": positive_count / len(frame),
                "actual_nonpositive_rate": (len(frame) - positive_count) / len(frame),
                "baseline_mae": baseline_mae,
                "mae_skill": np.nan if baseline_mae == 0.0 else 1.0 - mae / baseline_mae,
                "baseline_directional_accuracy": baseline_da,
                "da_difference": directional_accuracy - baseline_da,
            }
        )
    recomputed_per_asset = pd.DataFrame(
        per_asset_rows, columns=PER_ASSET_METRIC_COLUMNS
    ).sort_values(["fold", "asset"], kind="mergesort", ignore_index=True)
    saved_per_asset = tables["per_asset_metrics"].sort_values(
        ["fold", "asset"], kind="mergesort", ignore_index=True
    )
    categorical_columns = [
        "run_id",
        "spec_version",
        "data_version",
        "model",
        "model_config_id",
        "fold",
        "asset",
    ]
    for column in categorical_columns:
        if not saved_per_asset[column].astype(str).equals(
            recomputed_per_asset[column].astype(str)
        ):
            raise Phase2EExecutionError(f"Per-asset categorical field {column} differs.")
    count_columns = ["seed", "n", "actual_positive_count", "actual_nonpositive_count"]
    for column in count_columns:
        if not np.array_equal(
            saved_per_asset[column].to_numpy(dtype="int64"),
            recomputed_per_asset[column].to_numpy(dtype="int64"),
        ):
            raise Phase2EExecutionError(f"Per-asset count field {column} differs.")
    metric_columns = [
        "mae",
        "rmse",
        "directional_accuracy",
        "actual_positive_rate",
        "actual_nonpositive_rate",
        "baseline_mae",
        "mae_skill",
        "baseline_directional_accuracy",
        "da_difference",
    ]
    maximum_differences: dict[str, float] = {}
    for column in metric_columns:
        maximum_differences[f"per_asset.{column}"] = _assert_close(
            saved_per_asset[column].to_numpy(dtype="float64"),
            recomputed_per_asset[column].to_numpy(dtype="float64"),
            f"per-asset {column}",
        )

    macro_rows: list[dict[str, Any]] = []
    for fold_id, frame in recomputed_per_asset.groupby("fold", sort=True):
        if len(frame) != 4 or set(frame["asset"]) != {"AAPL", "MSFT", "GOOGL", "NVDA"}:
            raise Phase2EExecutionError(f"Recomputed macro fold {fold_id} is incomplete.")
        first = frame.iloc[0]
        macro_rows.append(
            {
                "run_id": first["run_id"],
                "spec_version": first["spec_version"],
                "data_version": first["data_version"],
                "model": "lightgbm",
                "model_config_id": first["model_config_id"],
                "seed": 1729,
                "fold": fold_id,
                "n_assets": 4,
                "n_observations": int(frame["n"].sum()),
                "mae": float(frame["mae"].mean()),
                "rmse": float(frame["rmse"].mean()),
                "directional_accuracy": float(frame["directional_accuracy"].mean()),
                "actual_positive_rate": float(frame["actual_positive_rate"].mean()),
                "actual_nonpositive_rate": float(frame["actual_nonpositive_rate"].mean()),
                "baseline_mae": float(frame["baseline_mae"].mean()),
                "mae_skill": float(frame["mae_skill"].mean()),
                "baseline_directional_accuracy": float(
                    frame["baseline_directional_accuracy"].mean()
                ),
                "da_difference": float(frame["da_difference"].mean()),
                "macro_weighting": "equal_weight_across_assets",
                "mae_skill_aggregation": "arithmetic_mean_of_asset_level_mae_skill",
            }
        )
    recomputed_macro = pd.DataFrame(macro_rows, columns=MACRO_METRIC_COLUMNS).sort_values(
        "fold", kind="mergesort", ignore_index=True
    )
    saved_macro = tables["macro_metrics"].sort_values(
        "fold", kind="mergesort", ignore_index=True
    )
    for column in (
        "run_id",
        "spec_version",
        "data_version",
        "model",
        "model_config_id",
        "fold",
        "macro_weighting",
        "mae_skill_aggregation",
    ):
        if not saved_macro[column].astype(str).equals(recomputed_macro[column].astype(str)):
            raise Phase2EExecutionError(f"Macro categorical field {column} differs.")
    for column in ("seed", "n_assets", "n_observations"):
        if not np.array_equal(
            saved_macro[column].to_numpy(dtype="int64"),
            recomputed_macro[column].to_numpy(dtype="int64"),
        ):
            raise Phase2EExecutionError(f"Macro count field {column} differs.")
    for column in metric_columns:
        maximum_differences[f"macro.{column}"] = _assert_close(
            saved_macro[column].to_numpy(dtype="float64"),
            recomputed_macro[column].to_numpy(dtype="float64"),
            f"macro {column}",
        )

    candidate_metrics = tables["candidate_metrics"]
    candidate_predictions = tables["candidate_predictions"]
    if tuple(candidate_metrics.columns) != CANDIDATE_METRIC_COLUMNS or len(
        candidate_metrics
    ) != 80:
        raise Phase2EExecutionError("Candidate audit table is incomplete or malformed.")
    if tuple(candidate_predictions.columns) != PREDICTION_COLUMNS:
        raise Phase2EExecutionError("Candidate prediction table has the wrong schema.")
    for column in ("origin_date", "target_date"):
        candidate_predictions[column] = pd.to_datetime(
            candidate_predictions[column], errors="raise"
        )
    if not candidate_predictions["partition"].eq("validation").all():
        raise Phase2EExecutionError("Candidate predictions contain non-validation rows.")
    if not candidate_metrics["status"].eq("completed").all() or not candidate_metrics[
        "error_status"
    ].eq("none").all():
        raise Phase2EExecutionError("Candidate audit contains incomplete or failed fits.")
    if candidate_metrics.duplicated(["fold", "model_config_id", "asset"]).any():
        raise Phase2EExecutionError("Candidate audit contains duplicate fits.")
    if not np.isfinite(
        candidate_metrics[
            [
                "validation_mae",
                "validation_rmse",
                "validation_directional_accuracy",
                "actual_positive_rate",
                "fit_runtime_seconds",
            ]
        ].to_numpy(dtype="float64")
    ).all():
        raise Phase2EExecutionError("Candidate audit contains non-finite required values.")
    for row in candidate_metrics.itertuples(index=False):
        frame = candidate_predictions.loc[
            (candidate_predictions["fold"] == row.fold)
            & (candidate_predictions["model_config_id"] == row.model_config_id)
            & (candidate_predictions["asset"] == row.asset)
        ]
        if len(frame) != int(row.n_validation):
            raise Phase2EExecutionError("Candidate prediction count disagrees with audit.")
        error = frame["predicted_log_return"].to_numpy(dtype="float64") - frame[
            "actual_log_return"
        ].to_numpy(dtype="float64")
        audit_values = np.array(
            [
                row.validation_mae,
                row.validation_rmse,
                row.validation_directional_accuracy,
                row.actual_positive_rate,
            ],
            dtype="float64",
        )
        recomputed_values = np.array(
            [
                np.mean(np.abs(error)),
                np.sqrt(np.mean(np.square(error))),
                np.mean(
                    frame["predicted_direction"].to_numpy(dtype="int8")
                    == frame["actual_direction"].to_numpy(dtype="int8")
                ),
                frame["actual_direction"].mean(),
            ],
            dtype="float64",
        )
        _assert_close(audit_values, recomputed_values, "candidate audit metrics")

    best_iterations = tables["best_iterations"]
    refit_audit = tables["refit_audit"]
    if tuple(best_iterations.columns) != BEST_ITERATION_COLUMNS or len(
        best_iterations
    ) != 20:
        raise Phase2EExecutionError("Winning best-iteration table is malformed.")
    if tuple(refit_audit.columns) != REFIT_AUDIT_COLUMNS or len(refit_audit) != 20:
        raise Phase2EExecutionError("Winner-refit audit table is malformed.")
    if not refit_audit["status"].eq("completed").all() or not refit_audit[
        "error_status"
    ].eq("none").all():
        raise Phase2EExecutionError("Winner-refit audit contains a failure.")
    iteration_match = refit_audit.merge(
        best_iterations.loc[
            :, ["fold", "model_config_id", "asset", "best_iteration"]
        ],
        on=["fold", "model_config_id", "asset"],
        how="left",
        validate="one_to_one",
    )
    if not np.array_equal(
        iteration_match["n_estimators"].to_numpy(dtype="int64"),
        iteration_match["best_iteration"].to_numpy(dtype="int64"),
    ):
        raise Phase2EExecutionError("Refit estimators differ from selected best iterations.")

    if len(artifacts.candidate_models) != 80 or len(artifacts.refit_models) != 20:
        raise Phase2EExecutionError("Saved fitted-model artifact count is incomplete.")
    for artifact in (*artifacts.candidate_models, *artifacts.refit_models):
        if file_sha256(artifact.path) != artifact.sha256:
            raise Phase2EExecutionError(f"Saved model artifact hash failed: {artifact.path}.")

    return {
        "status": "passed",
        "verification_method": "independent_reload_and_formula_recomputation",
        "prediction_rows": len(predictions),
        "canonical_column_count": len(predictions.columns),
        "baseline_rows": len(baseline),
        "exact_baseline_key_match": True,
        "duplicate_prediction_keys": 0,
        "development_folds": list(DEVELOPMENT_FOLD_IDS),
        "target_years": [2020, 2021, 2022, 2023, 2024],
        "f1_rows": 0,
        "partition_values": ["test"],
        "seed_values": [1729],
        "model_values": ["lightgbm"],
        "winner_configuration_match": True,
        "finite_predictions": True,
        "sign_derived_direction_match": True,
        "regime_sentinel_only": True,
        "candidate_audit_rows": len(candidate_metrics),
        "candidate_audit_fields_complete": True,
        "winning_best_iteration_rows": len(best_iterations),
        "winner_refit_audit_rows": len(refit_audit),
        "candidate_model_artifacts": len(artifacts.candidate_models),
        "refit_model_artifacts": len(artifacts.refit_models),
        "metric_float_rtol": FLOAT_RTOL,
        "metric_float_atol": FLOAT_ATOL,
        "maximum_absolute_differences": maximum_differences,
        "per_asset_metrics_recomputed": True,
        "macro_metrics_recomputed": True,
        "macro_mae_skill_rule": "arithmetic_mean_of_asset_level_mae_skill",
    }


def _artifact_record(artifact: Any) -> dict[str, Any]:
    return {
        "path": _relative(artifact.path),
        "sha256": artifact.sha256,
        "byte_count": artifact.byte_count,
    }


def _fold_bounds_and_counts(dry_run: Mapping[str, Any]) -> dict[str, Any]:
    registry = load_fold_registry()
    result: dict[str, Any] = {}
    for fold_id in DEVELOPMENT_FOLD_IDS:
        fold = registry.get(fold_id)
        result[fold_id] = {
            "train": {
                "start": fold.train.start.isoformat(),
                "end": fold.train.end.isoformat(),
                **dry_run["fold_partition_counts"][fold_id]["train"],
            },
            "validation": {
                "start": fold.validation.start.isoformat(),
                "end": fold.validation.end.isoformat(),
                **dry_run["fold_partition_counts"][fold_id]["validation"],
            },
            "test": {
                "start": fold.test.start.isoformat(),
                "end": fold.test.end.isoformat(),
                **dry_run["fold_partition_counts"][fold_id]["test"],
            },
        }
    return result


def build_run_manifest(
    *,
    verified_inputs: Mapping[str, Any],
    environment: Mapping[str, Any],
    dry_run: Mapping[str, Any],
    run: LightGBMDevelopmentRun,
    artifacts: LightGBMArtifacts,
    reproducibility_artifact: Any,
    reproducibility_result: Mapping[str, Any],
    verification_artifact: Any,
    verification_result: Mapping[str, Any],
    started_at: datetime,
    completed_at: datetime,
    runtime_seconds: float,
) -> dict[str, Any]:
    contract: LightGBMContract = verified_inputs["contract"]
    selected_rows = run.selections.loc[run.selections["selected"]]
    winners = {
        str(row.fold): str(row.model_config_id)
        for row in selected_rows.itertuples(index=False)
    }
    best_iterations = {
        fold_id: {
            str(row.asset): int(row.best_iteration)
            for row in run.best_iterations.loc[
                run.best_iterations["fold"] == fold_id
            ].itertuples(index=False)
        }
        for fold_id in DEVELOPMENT_FOLD_IDS
    }

    candidate_statuses = []
    for result, artifact in zip(
        run.candidate_results, artifacts.candidate_models, strict=True
    ):
        candidate_statuses.append(
            {
                "fold": result.fold,
                "config_id": result.candidate.config_id,
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
                "warning_messages": list(result.warning_messages),
                "error_status": result.error_status,
                "error_type": result.error_type,
                "error_message": result.error_message,
                "effective_parameters": resolve_estimator_parameters(
                    contract, result.candidate
                ),
                "input_scaling": "none",
                "model_artifact": _artifact_record(artifact),
            }
        )

    refit_statuses = []
    for result, artifact in zip(run.refit_results, artifacts.refit_models, strict=True):
        refit_statuses.append(
            {
                "fold": result.fold,
                "config_id": result.candidate.config_id,
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
                "warning_messages": list(result.warning_messages),
                "error_status": result.error_status,
                "error_type": result.error_type,
                "error_message": result.error_message,
                "fresh_model": True,
                "training_rows": "train_plus_validation",
                "early_stopping": False,
                "eval_set": None,
                "callbacks": None,
                "effective_parameters": resolve_estimator_parameters(
                    contract,
                    result.candidate,
                    n_estimators=result.n_estimators,
                ),
                "input_scaling": "none",
                "model_artifact": _artifact_record(artifact),
            }
        )

    all_warning_messages = sorted(
        {
            message
            for result in (*run.candidate_results, *run.refit_results)
            for message in result.warning_messages
        }
    )
    return {
        "schema_version": 1,
        "artifact_type": "phase2e_lightgbm_run_manifest",
        "run_id": RUN_ID,
        "status": "completed_and_independently_verified",
        "created_at_utc": started_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "completed_at_utc": completed_at.astimezone(UTC).isoformat().replace(
            "+00:00", "Z"
        ),
        "experiment_specification": {
            "version": contract.spec_version,
            "path": "docs/EXPERIMENT_SPEC.md",
            "sha256": file_sha256(REPOSITORY_ROOT / "docs" / "EXPERIMENT_SPEC.md"),
        },
        "model_preregistration": {
            "path": _relative(PREREGISTRATION_PATH),
            "sha256": PREREGISTRATION_SHA256,
        },
        "model_search": {
            "path": _relative(MODEL_SEARCH_PATH),
            "sha256": MODEL_SEARCH_SHA256,
        },
        "inputs": {
            "raw_dataset": {
                "path": _relative(RAW_DATA_PATH),
                "manifest_path": _relative(RAW_MANIFEST_PATH),
                "sha256": RAW_SHA256,
            },
            "processed_dataset": {
                "path": _relative(PROCESSED_DATA_PATH),
                "manifest_path": _relative(PROCESSED_MANIFEST_PATH),
                "sha256": PROCESSED_SHA256,
            },
            "canonical_phase2c_baseline": {
                "path": _relative(BASELINE_PATH),
                "sha256": BASELINE_SHA256,
                "rows": EXPECTED_PREDICTION_ROWS,
                "discovery_method": "exact_pinned_path_and_sha256_not_glob",
            },
        },
        "source_revision": _source_tree_identity(),
        "feature_contract": {
            "count": len(contract.feature_columns),
            "ordered_columns": list(contract.feature_columns),
            "input_scaling": "none",
        },
        "candidate_definitions": [
            {
                "config_id": candidate.config_id,
                "num_leaves": candidate.num_leaves,
                "min_child_samples": candidate.min_child_samples,
                "effective_parameters": resolve_estimator_parameters(contract, candidate),
            }
            for candidate in contract.candidates
        ],
        "selection": {
            "scope": "one_shared_structural_winner_per_fold",
            "primary_statistic": "equal_weight_macro_validation_mae",
            "tie_tolerance": contract.tie_tolerance,
            "tie_rules": [
                "structural_simplicity",
                "equal_weight_macro_validation_rmse",
                "ascending_lexical_config_id",
            ],
            "winner_by_fold": winners,
            "best_iteration_by_fold_and_asset": best_iterations,
            "directional_accuracy_used_for_selection": False,
            "runtime_used_for_selection": False,
            "test_metrics_used_for_selection": False,
            "future_folds_used_for_selection": False,
            "regime_information_used_for_selection": False,
        },
        "fit_accounting": {
            "candidate_fits_completed": len(run.candidate_results),
            "candidate_fits_expected": contract.max_candidate_fits,
            "winner_refits_completed": len(run.refit_results),
            "winner_refits_expected": contract.max_winning_refits,
            "bounded_reproducibility_check_fits": 2,
            "candidate_statuses": candidate_statuses,
            "refit_statuses": refit_statuses,
        },
        "environment": {
            "manifest_path": environment["manifest_path"],
            "manifest_sha256": environment["manifest_sha256"],
            "packages": environment["packages"],
            "python_version": environment["python_version"],
            "python_executable": environment["python_executable"],
            "lightgbm_module_path": environment["lightgbm_module_path"],
            "device_policy": environment["device_policy"],
            "candidate_parallelism": "sequential",
            "refit_parallelism": "sequential",
            "lightgbm_n_jobs": 1,
        },
        "determinism": {
            "random_state": 1729,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": 1,
            "reference_execution": "cpu_only_sequential",
        },
        "folds": _fold_bounds_and_counts(dry_run),
        "prediction_contract": {
            "columns": list(PREDICTION_COLUMNS),
            "row_count": len(run.predictions),
            "partition": "test",
            "folds": list(DEVELOPMENT_FOLD_IDS),
            "target_years": [2020, 2021, 2022, 2023, 2024],
            "model": "lightgbm",
            "seed": 1729,
            "predicted_direction_rule": "1 if predicted_log_return > 0 else 0",
            "regime_placeholder": REGIME_PLACEHOLDER,
            "exact_baseline_key_match": True,
        },
        "metrics": {
            "candidate_selection": ["validation_mae", "validation_rmse_tie_only"],
            "test": [
                "mae",
                "rmse",
                "directional_accuracy",
                "actual_positive_rate",
                "baseline_mae",
                "mae_skill",
                "baseline_directional_accuracy",
                "da_difference",
            ],
            "macro_weighting": "equal_weight_across_assets",
            "macro_mae_skill": "arithmetic_mean_of_four_asset_level_skills",
            "development_summary": "median_q1_q3_iqr_linear_interpolation_descriptive_only",
        },
        "artifacts": {
            "environment": {
                "path": environment["manifest_path"],
                "sha256": environment["manifest_sha256"],
            },
            "tables": {
                name: _artifact_record(artifact)
                for name, artifact in artifacts.tables.items()
            },
            "candidate_models": [
                _artifact_record(artifact) for artifact in artifacts.candidate_models
            ],
            "winner_refit_models": [
                _artifact_record(artifact) for artifact in artifacts.refit_models
            ],
            "reproducibility_check": _artifact_record(reproducibility_artifact),
            "independent_verification": _artifact_record(verification_artifact),
        },
        "reproducibility_check": dict(reproducibility_result),
        "independent_verification": dict(verification_result),
        "runtime_provenance": {
            "command": "python -m src.evaluation.run_lightgbm",
            "working_directory": str(REPOSITORY_ROOT.resolve()),
            "runtime_seconds": runtime_seconds,
            "host": platform.node(),
            "platform": platform.platform(),
            "candidate_and_refit_execution": "sequential",
        },
        "scope_guards": {
            "development_folds_only": list(DEVELOPMENT_FOLD_IDS),
            "f1_status": "LOCKED_UNTOUCHED",
            "f1_candidate_fit_or_selection_performed": False,
            "target_2025_performance_evaluated": False,
            "regime_analysis_status": "not_performed_placeholders_only",
            "lstm_implemented_or_trained": False,
            "patchtst_status": "CLOSED_NOT_IMPLEMENTED",
        },
        "warnings": all_warning_messages,
        "exceptions": [],
        "specification_deviations": [],
        "dry_run_gate": dict(dry_run),
    }


def execute_benchmark(
    verified_inputs: Mapping[str, Any],
    environment: Mapping[str, Any],
    dry_run: Mapping[str, Any],
) -> Path:
    """Execute exactly 80 candidates, 20 refits, checks, and one manifest."""

    _assert_only_environment_exists()
    contract: LightGBMContract = verified_inputs["contract"]
    started_at = datetime.now(UTC)
    timer = perf_counter()
    try:
        run = run_development_lightgbm(
            verified_inputs["samples"],
            verified_inputs["baseline"],
            run_id=RUN_ID,
            data_version=PROCESSED_SHA256,
            contract=contract,
            progress_callback=_progress,
        )
        if len(run.candidate_results) != 80 or len(run.refit_results) != 20:
            raise Phase2EExecutionError("Real benchmark fit accounting is incomplete.")
        artifacts = write_development_artifacts(run, output_root=OUTPUT_ROOT)

        reproducibility_result = run_bounded_reproducibility_check(contract)
        reproducibility_artifact = write_content_addressed_bytes(
            OUTPUT_ROOT / "verification",
            prefix="lightgbm_reproducibility_check",
            suffix=".json",
            payload=canonical_json_bytes(reproducibility_result),
        )
        verification_result = independently_verify_saved_results(
            verified_inputs, artifacts
        )
        verification_artifact = write_content_addressed_bytes(
            OUTPUT_ROOT / "verification",
            prefix="lightgbm_independent_verification",
            suffix=".json",
            payload=canonical_json_bytes(verification_result),
        )

        _require_hash(RAW_DATA_PATH, RAW_SHA256, "Post-run raw dataset")
        _require_hash(PROCESSED_DATA_PATH, PROCESSED_SHA256, "Post-run processed dataset")
        _require_hash(
            PREREGISTRATION_PATH, PREREGISTRATION_SHA256, "Post-run preregistration"
        )
        _require_hash(MODEL_SEARCH_PATH, MODEL_SEARCH_SHA256, "Post-run model search")
        _require_hash(ENVIRONMENT_PATH, ENVIRONMENT_SHA256, "Post-run environment lock")

        completed_at = datetime.now(UTC)
        elapsed = perf_counter() - timer
        manifest = build_run_manifest(
            verified_inputs=verified_inputs,
            environment=environment,
            dry_run=dry_run,
            run=run,
            artifacts=artifacts,
            reproducibility_artifact=reproducibility_artifact,
            reproducibility_result=reproducibility_result,
            verification_artifact=verification_artifact,
            verification_result=verification_result,
            started_at=started_at,
            completed_at=completed_at,
            runtime_seconds=elapsed,
        )
        manifest_artifact = write_content_addressed_bytes(
            OUTPUT_ROOT,
            prefix="run_manifest",
            suffix=".json",
            payload=canonical_json_bytes(manifest),
        )
        print(f"Created Phase 2E run manifest: {manifest_artifact.path}", flush=True)
        print(
            f"Verified canonical predictions: {artifacts.tables['predictions'].path}",
            flush=True,
        )
        return manifest_artifact.path
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "artifact_type": "phase2e_lightgbm_failure_ledger",
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "status": "failed_stop_without_incomplete_selection",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "rerun_policy": "do_not_select_from_or_continue_an_incomplete_grid",
        }
        write_content_addressed_bytes(
            OUTPUT_ROOT / "failures",
            prefix="failure_ledger",
            suffix=".json",
            payload=canonical_json_bytes(failure),
        )
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 2E LightGBM benchmark on D1--D5 only."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all gates and counts without constructing or fitting a model.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    contract = load_lightgbm_contract(MODEL_SEARCH_PATH)
    environment = verify_exact_environment()
    verified_inputs = verify_frozen_inputs(contract)
    dry_run = perform_dry_run(verified_inputs, environment)
    if args.dry_run:
        _assert_only_environment_exists()
        print(json.dumps(dry_run, sort_keys=True, indent=2), flush=True)
        return 0
    execute_benchmark(verified_inputs, environment, dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
