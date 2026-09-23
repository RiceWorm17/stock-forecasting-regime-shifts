"""Run only the preregistered Phase 2H-B D1 train/validation PatchTST pilot.

Allowed public commands::

    python -m src.evaluation.run_patchtst_pilot --dry-run
    python -m src.evaluation.run_patchtst_pilot

There is deliberately no fold selector, test path, year override, model-selection
path, full-benchmark path, F1 path, or regime-analysis path.
"""

from __future__ import annotations

import argparse
import ctypes
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
from src.evaluation.patchtst_pilot import (
    D1_TRAIN_END,
    D1_TRAIN_START,
    D1_VALIDATION_END,
    D1_VALIDATION_START,
    EXPECTED_CANDIDATE_FITS,
    EXPECTED_REPLAY_EXECUTIONS,
    EXPECTED_SCALERS,
    EXPECTED_TOTAL_TRAINING_EXECUTIONS,
    FIT_AUDIT_COLUMNS,
    PILOT_FOLD,
    PILOT_PARTITION,
    CandidateFitResult,
    PatchTSTPilotError,
    ScalerRecord,
    SequenceSet,
    WrittenArtifact,
    build_gate_decision,
    build_sequences,
    candidate_results_frames,
    canonical_csv_bytes,
    canonical_json_bytes,
    compute_resource_feasibility,
    fit_candidate,
    fit_candidate_scaler,
    independently_verify_saved_pilot,
    regression_metrics,
    sha256_bytes,
    split_d1_train_validation,
    validate_complete_candidate_grid,
    validate_pilot_samples,
    validation_coverage_audit,
    verify_deterministic_replay,
    write_content_addressed,
)
from src.models.patchtst_model import (
    FROZEN_CANDIDATES,
    FROZEN_FEATURE_COLUMNS,
    FUTURE_FULL_SEEDS,
    PILOT_SEEDS,
    TARGET_ASSETS,
    PatchTSTContract,
    configure_cpu_runtime,
    load_patchtst_contract,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "patchtst" / "pilot"

PATCHTST_PREREGISTRATION_SHA256 = (
    "adb4be727379d005dc69fd6d8528c6e46abda993394f21a946097e59cabfb486"
)
PATCHTST_SEARCH_SHA256 = (
    "6be2b9e2017c1f7b89c553ef1f65eaa995e6cd2cbc233fa2cd84a51a12c91124"
)
PHASE_2H_A_REPORT_SHA256 = (
    "72d80bdc0725682754a3d939070865d7d96564a290919b8f1a6bf05442321f95"
)
RAW_SHA256 = "55f74f058493102387f1c2d0a848bd673a3cfa0473d617dfdb34aea8a94f7736"
PROCESSED_SHA256 = "9352da6b434011f9844ef0acc1813ea33dce39152712752bba21a3005aabf7be"
PYPROJECT_SHA256 = "7fc8cf9c6fbc6440061cc68f3a515df28b521c724b32f345a5447f7844226309"
PHASE_2G_REPORT_SHA256 = (
    "e745662a3a8a8a8497348e2480e821a44ab4b55d7ff16721e3f42067db607667"
)
PHASE_2G_GATE_SHA256 = (
    "b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7"
)
PHASE_2G_VERIFICATION_SHA256 = (
    "393be15393847bdcc48d9482a0d72f32437f1024e85523088dbc3194c4eabb02"
)
FROZEN_FILE_HASHES: Mapping[str, str] = {
    "docs/EXPERIMENT_SPEC.md": "4a29b96a7c826634d2dede63220c378737c0e545a7e733491a0a7002d69fc5b9",
    "docs/DATA_CONTRACT.md": "838cbcb2419a14fe413c3b93036b5daa29a97c6f338b52dcc07376116ed2263d",
    "docs/MODEL_PREREGISTRATION.md": "be04b9ce8457c5b26a8042f548eccce8ddb4b5d1b86a1404a43ef1bb1a52d613",
    "configs/data.yaml": "bdbd3d48be74e09228d500c0e753af0da7e41cac43055e1652f897855e0bd3ed",
    "configs/walk_forward.yaml": "ac709ec4bb76e9e963ddeee072430b3511b46457846715e11f634a0f13d37d08",
    "configs/model_search.yaml": "e15e42ff2b2f4f2bc7ccc185195c26fd35851b259bc28e08a6ba2149b6aaf38c",
    "docs/PATCHTST_PREREGISTRATION.md": PATCHTST_PREREGISTRATION_SHA256,
    "configs/patchtst_search.yaml": PATCHTST_SEARCH_SHA256,
    "PHASE_2H_A_REPORT.md": PHASE_2H_A_REPORT_SHA256,
    "docs/audit/PHASE_2G_REPORT.md": PHASE_2G_REPORT_SHA256,
    "results/combined/patchtst_gate_decision_b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7.json": PHASE_2G_GATE_SHA256,
    "results/combined/combined_verification_manifest_393be15393847bdcc48d9482a0d72f32437f1024e85523088dbc3194c4eabb02.json": PHASE_2G_VERIFICATION_SHA256,
    "pyproject.toml": PYPROJECT_SHA256,
}
FROZEN_TREE_PATHS: Mapping[str, str] = {
    "baseline": "results/baselines",
    "lightgbm": "results/lightgbm",
    "lstm": "results/lstm",
    "combined_phase2g": "results/combined",
}

RAW_DATA_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "yfinance_daily_2015-01-01_2025-12-31_20260913T152313Z_55f74f058493.csv"
)
PROCESSED_DATA_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / f"supervised_v1_{PROCESSED_SHA256}.csv"
)
PATCHTST_PREREGISTRATION_PATH = REPOSITORY_ROOT / "docs" / "PATCHTST_PREREGISTRATION.md"
PATCHTST_SEARCH_PATH = REPOSITORY_ROOT / "configs" / "patchtst_search.yaml"
PYPROJECT_PATH = REPOSITORY_ROOT / "pyproject.toml"
RUN_ID = f"phase2hb_patchtst_pilot_{PROCESSED_SHA256[:12]}"
FINALIZATION_ALLOWANCE_SECONDS = 5.0


class Phase2HBExecutionError(RuntimeError):
    """Raised when a frozen Phase 2H-B execution invariant fails."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _require_hash(path: Path, expected: str, label: str) -> str:
    observed = file_sha256(path)
    if observed != expected:
        raise Phase2HBExecutionError(
            f"{label} SHA-256 mismatch: expected={expected}, observed={observed}."
        )
    return observed


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise Phase2HBExecutionError(f"Required distribution {name!r} is unavailable.") from exc


def tree_identity(relative_root: str) -> dict[str, Any]:
    """Reproduce the frozen Phase 2G tree-identity algorithm exactly."""

    root = (REPOSITORY_ROOT / relative_root).resolve()
    if not root.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise Phase2HBExecutionError(f"Tree escaped repository: {relative_root}.")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
    )
    records = [
        f"{path.relative_to(REPOSITORY_ROOT).as_posix()}\t{file_sha256(path)}"
        for path in files
    ]
    return {
        "path": relative_root,
        "method": "sha256_of_sorted_repo_relative_path_tab_file_sha256_lf_records",
        "file_count": len(files),
        "sha256": hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest(),
    }


def source_tree_identity() -> dict[str, Any]:
    files = sorted(
        (path for path in (REPOSITORY_ROOT / "src").rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
    )
    digest = hashlib.sha256()
    per_file: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        value = file_sha256(path)
        per_file[relative] = value
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return {
        "method": "sha256_of_sorted_relative_path_nul_file_sha256_lf_records",
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": per_file,
        "git_revision": None,
        "git_status": "not_available_repository_has_no_git_metadata",
    }


def verify_frozen_identities(contract: PatchTSTContract) -> dict[str, Any]:
    """Verify all frozen files and historical trees without reading performance."""

    _require_hash(RAW_DATA_PATH, RAW_SHA256, "Raw dataset")
    _require_hash(PROCESSED_DATA_PATH, PROCESSED_SHA256, "Processed dataset")
    file_records: dict[str, str] = {}
    for relative, expected in FROZEN_FILE_HASHES.items():
        file_records[relative] = _require_hash(
            REPOSITORY_ROOT / relative, expected, f"Frozen file {relative}"
        )
    if contract.raw_sha256 != RAW_SHA256 or contract.processed_sha256 != PROCESSED_SHA256:
        raise Phase2HBExecutionError("PatchTST contract contains unexpected data identities.")
    observed_trees: dict[str, dict[str, Any]] = {}
    for key, relative in FROZEN_TREE_PATHS.items():
        observed = tree_identity(relative)
        expected = contract.frozen_trees[key]
        if observed["file_count"] != int(expected["file_count"]) or observed["sha256"] != str(
            expected["sha256"]
        ):
            raise Phase2HBExecutionError(
                f"Frozen {key} tree drifted: expected={dict(expected)!r}, observed={observed!r}."
            )
        observed_trees[key] = observed
    return {
        "status": "PASS",
        "raw_dataset_sha256": RAW_SHA256,
        "processed_dataset_sha256": PROCESSED_SHA256,
        "frozen_files": file_records,
        "frozen_trees": observed_trees,
        "f1_performance_accessed": False,
        "d2_d5_model_performance_accessed": False,
        "regime_analysis_performed": False,
    }


def _current_process_peak_memory() -> dict[str, Any]:
    """Read Windows peak working-set bytes without adding a dependency."""

    if platform.system() != "Windows":
        return {
            "status": "unavailable",
            "method": None,
            "peak_process_memory_bytes": None,
            "reason": "Windows PSAPI measurement is unavailable on this platform.",
        }
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo returned false")
        return {
            "status": "recorded",
            "method": "Windows_PSAPI_PeakWorkingSetSize",
            "peak_process_memory_bytes": int(counters.PeakWorkingSetSize),
            "reason": None,
        }
    except Exception as exc:  # pragma: no cover - platform failure is documented
        return {
            "status": "unavailable",
            "method": "Windows_PSAPI_PeakWorkingSetSize",
            "peak_process_memory_bytes": None,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def capture_environment_manifest(contract: PatchTSTContract) -> tuple[dict[str, Any], WrittenArtifact]:
    """Freeze and persist the exact environment before any first pilot fit."""

    runtime_policy = configure_cpu_runtime()
    observed_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": _distribution_version("scikit-learn"),
        "pyyaml": _distribution_version("PyYAML"),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    expected = dict(contract.environment_versions)
    for key in ("python", "torch", "numpy", "pandas", "scikit_learn", "pyyaml"):
        if str(observed_versions[key]) != str(expected[key]):
            raise Phase2HBExecutionError(
                f"Environment {key} drifted: expected={expected[key]!r}, observed={observed_versions[key]!r}."
            )
    if observed_versions["cuda_available"] is not False or expected["cuda_available"] is not False:
        raise Phase2HBExecutionError("The reference pilot must run with CUDA unavailable.")
    expected_policy = {
        "device": "cpu",
        "intra_op_threads": 1,
        "inter_op_threads": 1,
        "deterministic_algorithms_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    if runtime_policy != expected_policy:
        raise Phase2HBExecutionError(f"CPU deterministic policy drifted: {runtime_policy!r}.")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "patchtst_pilot_environment",
        "captured_before_first_fit": True,
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "compiler": platform.python_compiler(),
        },
        "packages": {
            "torch_runtime": torch.__version__,
            "torch_distribution": _distribution_version("torch"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": _distribution_version("scikit-learn"),
            "pyyaml": _distribution_version("PyYAML"),
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "python_architecture": platform.architecture()[0],
        },
        "device": {
            "policy": "cpu_only",
            "execution_device": "cpu",
            "cuda_available": False,
            "cuda_version": torch.version.cuda,
        },
        "runtime_policy": runtime_policy,
        "repository_inputs": {
            "pyproject_path": "pyproject.toml",
            "pyproject_sha256": _require_hash(PYPROJECT_PATH, PYPROJECT_SHA256, "pyproject"),
            "patchtst_preregistration_path": "docs/PATCHTST_PREREGISTRATION.md",
            "patchtst_preregistration_sha256": _require_hash(
                PATCHTST_PREREGISTRATION_PATH,
                PATCHTST_PREREGISTRATION_SHA256,
                "PatchTST preregistration",
            ),
            "patchtst_search_path": "configs/patchtst_search.yaml",
            "patchtst_search_sha256": _require_hash(
                PATCHTST_SEARCH_PATH, PATCHTST_SEARCH_SHA256, "PatchTST search"
            ),
        },
        "source_tree_identity": source_tree_identity(),
    }
    artifact = write_content_addressed(
        OUTPUT_ROOT / "environment",
        "patchtst_pilot_environment",
        ".json",
        canonical_json_bytes(manifest),
    )
    return manifest, artifact


def load_d1_pilot_rows(contract: PatchTSTContract) -> tuple[pd.DataFrame, int]:
    """Load the frozen samples and discard every post-2019 row before validation."""

    columns = [
        "asset",
        "origin_date",
        "target_date",
        "target_log_return",
        "target_direction",
        "core_evaluation_eligible",
        *contract.feature_columns,
    ]
    frame = pd.read_csv(PROCESSED_DATA_PATH, usecols=columns)
    target_dates = pd.to_datetime(frame["target_date"], errors="raise")
    authorized = target_dates.between(D1_TRAIN_START, D1_VALIDATION_END, inclusive="both")
    excluded_count = int((~authorized).sum())
    pilot = frame.loc[authorized].copy()
    pilot["target_date"] = target_dates.loc[authorized]
    if (pilot["target_date"].dt.year >= 2020).any():
        raise Phase2HBExecutionError("Post-2019 target escaped the pilot load boundary.")
    return pilot, excluded_count


def _asset_rows(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    result = frame.loc[frame["asset"] == asset].sort_values(
        "origin_date", kind="mergesort", ignore_index=True
    )
    if result.empty:
        raise Phase2HBExecutionError(f"Missing required asset slice {asset}.")
    return result


def _key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        frame[["asset", "origin_date", "target_date"]].sort_values(
            ["asset", "origin_date", "target_date"], kind="mergesort", ignore_index=True
        )
    )


def prepare_pilot_objects(samples: pd.DataFrame, *, contract: PatchTSTContract) -> dict[str, Any]:
    """Fit four in-memory scalers and prove exact D1 validation constructibility."""

    normalized = validate_pilot_samples(samples, contract=contract)
    train, validation = split_d1_train_validation(normalized)
    scalers: dict[str, ScalerRecord] = {}
    sequences: dict[tuple[str, str, str], SequenceSet] = {}
    authoritative_validation: dict[str, pd.DataFrame] = {}
    endpoint_counts: list[dict[str, Any]] = []
    for asset in TARGET_ASSETS:
        asset_train = _asset_rows(train, asset)
        asset_validation = _asset_rows(validation, asset)
        asset_history = _asset_rows(normalized, asset)
        if len(asset_validation) != 252:
            raise Phase2HBExecutionError(
                f"Frozen D1 validation count for {asset} must be 252, got {len(asset_validation)}."
            )
        authoritative = asset_validation[
            ["asset", "origin_date", "target_date", "target_log_return", "target_direction"]
        ].copy()
        authoritative_validation[asset] = authoritative
        scaler = fit_candidate_scaler(
            asset_train,
            asset=asset,
            feature_columns=contract.feature_columns,
            processed_data_sha256=PROCESSED_SHA256,
            patchtst_config_sha256=PATCHTST_SEARCH_SHA256,
        )
        scalers[asset] = scaler
        authoritative_keys = _key_index(authoritative)
        for candidate in FROZEN_CANDIDATES:
            train_sequences = build_sequences(
                asset_history,
                asset_train,
                context_length=candidate.context_length,
                scaler=scaler.scaler,
                require_all_endpoints=False,
                partition_name="D1_train",
            )
            validation_sequences = build_sequences(
                asset_history,
                asset_validation,
                context_length=candidate.context_length,
                scaler=scaler.scaler,
                require_all_endpoints=True,
                partition_name="D1_validation",
            )
            if not _key_index(validation_sequences.rows).equals(authoritative_keys):
                raise Phase2HBExecutionError(
                    f"{candidate.config_id}/{asset} does not cover authoritative D1 validation keys."
                )
            sequences[(candidate.config_id, asset, "train")] = train_sequences
            sequences[(candidate.config_id, asset, "validation")] = validation_sequences
            endpoint_counts.append(
                {
                    "model_config_id": candidate.config_id,
                    "context_length": candidate.context_length,
                    "asset": asset,
                    "candidate_scaler_rows": scaler.row_count,
                    "train_endpoints": len(train_sequences.targets),
                    "train_endpoints_omitted_for_warmup": len(train_sequences.unavailable_keys),
                    "validation_endpoints": len(validation_sequences.targets),
                    "authoritative_validation_endpoints": len(authoritative),
                }
            )
    if len(scalers) != EXPECTED_SCALERS:
        raise Phase2HBExecutionError(f"Pilot must construct exactly {EXPECTED_SCALERS} scalers.")
    return {
        "normalized": normalized,
        "train": train,
        "validation": validation,
        "scalers": scalers,
        "sequences": sequences,
        "authoritative_validation": authoritative_validation,
        "endpoint_counts": endpoint_counts,
    }


def execution_plan() -> dict[str, Any]:
    """Return exactly 16 candidate identities plus one separate P5 replay."""

    candidate_fits = [
        {
            "model_config_id": candidate.config_id,
            "context_length": candidate.context_length,
            "asset": asset,
            "seed": seed,
        }
        for candidate in FROZEN_CANDIDATES
        for asset in TARGET_ASSETS
        for seed in PILOT_SEEDS
    ]
    replay = {
        "model_config_id": "PATCHTST_63",
        "context_length": 63,
        "asset": "AAPL",
        "seed": 1729,
        "role": "P5_non_candidate_verification_replay",
    }
    if len(candidate_fits) != EXPECTED_CANDIDATE_FITS or len(
        {
            (item["model_config_id"], item["asset"], item["seed"])
            for item in candidate_fits
        }
    ) != EXPECTED_CANDIDATE_FITS:
        raise AssertionError("Pilot execution plan is not the exact 16-fit grid.")
    return {
        "candidate_fits": candidate_fits,
        "candidate_fit_count": len(candidate_fits),
        "replay": replay,
        "replay_execution_count": EXPECTED_REPLAY_EXECUTIONS,
        "total_training_executions": EXPECTED_TOTAL_TRAINING_EXECUTIONS,
        "refits": 0,
        "test_predictions": 0,
    }


def perform_dry_run(
    *,
    contract: PatchTSTContract,
    environment_manifest: Mapping[str, Any],
    environment_artifact: WrittenArtifact,
    frozen: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, excluded = load_d1_pilot_rows(contract)
    prepared = prepare_pilot_objects(rows, contract=contract)
    plan = execution_plan()
    dry_run = {
        "schema_version": 1,
        "artifact_type": "patchtst_pilot_dry_run",
        "status": "PASS_NO_MODEL_FITTING_PERFORMED",
        "run_id": RUN_ID,
        "frozen_identity_status": frozen["status"],
        "environment_manifest": {
            "path": _relative(environment_artifact.path),
            "sha256": environment_artifact.sha256,
            "captured_before_first_fit": environment_manifest["captured_before_first_fit"],
        },
        "source_tree_sha256": environment_manifest["source_tree_identity"]["sha256"],
        "execution_device": "cpu",
        "cuda_available": False,
        "thread_policy": environment_manifest["runtime_policy"],
        "feature_count": len(contract.feature_columns),
        "feature_order": list(contract.feature_columns),
        "target_assets": list(TARGET_ASSETS),
        "market_reference": "SPY_feature_only",
        "candidates": [
            {
                "config_id": item.config_id,
                "context_length": item.context_length,
                "patch_start_indices": list(item.patch_start_indices),
                "patch_count": item.patch_count,
                "head_input_dimension": item.head_input_dimension,
            }
            for item in FROZEN_CANDIDATES
        ],
        "pilot_seeds": list(PILOT_SEEDS),
        "future_full_seeds_not_used": list(FUTURE_FULL_SEEDS),
        "assets": list(TARGET_ASSETS),
        "fold": PILOT_FOLD,
        "allowed_partitions": ["train", "validation"],
        "train_target_date_range": [D1_TRAIN_START.date().isoformat(), D1_TRAIN_END.date().isoformat()],
        "validation_target_date_range": [
            D1_VALIDATION_START.date().isoformat(),
            D1_VALIDATION_END.date().isoformat(),
        ],
        "post_2019_rows_excluded_before_model_validation": excluded,
        "d1_test_accessible": False,
        "d2_d5_performance_accessible": False,
        "f1_performance_accessible": False,
        "candidate_scalers_expected": EXPECTED_SCALERS,
        "candidate_scalers_constructed_in_memory": len(prepared["scalers"]),
        "endpoint_counts": prepared["endpoint_counts"],
        "validation_key_equality_and_authoritative_coverage": True,
        "candidate_fits_expected": plan["candidate_fit_count"],
        "replay_executions_expected": plan["replay_execution_count"],
        "total_training_executions_expected": plan["total_training_executions"],
        "refits_expected": plan["refits"],
        "development_test_predictions_expected": plan["test_predictions"],
        "model_fits_performed": 0,
        "performance_calculated": False,
    }
    return dry_run, prepared


def _write_dry_run_receipt(dry_run: Mapping[str, Any]) -> WrittenArtifact:
    return write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_pilot_dry_run",
        ".json",
        canonical_json_bytes(dry_run),
    )


def _require_prior_dry_run(dry_run: Mapping[str, Any]) -> WrittenArtifact:
    payload = canonical_json_bytes(dry_run)
    digest = sha256_bytes(payload)
    path = OUTPUT_ROOT / "verification" / f"patchtst_pilot_dry_run_{digest}.json"
    if not path.is_file() or file_sha256(path) != digest:
        raise Phase2HBExecutionError(
            "The exact zero-fit --dry-run receipt is required before pilot training."
        )
    return WrittenArtifact(path=path, sha256=digest)


def _assert_prefit_artifacts_only(
    environment_artifact: WrittenArtifact, dry_run_artifact: WrittenArtifact
) -> None:
    files = sorted(path.resolve() for path in OUTPUT_ROOT.rglob("*") if path.is_file())
    allowed = sorted([environment_artifact.path.resolve(), dry_run_artifact.path.resolve()])
    if files != allowed:
        unexpected = [_relative(path) for path in files if path not in allowed]
        raise Phase2HBExecutionError(
            f"Refusing to mix or overwrite a prior pilot; unexpected artifacts: {unexpected}."
        )


def _write_scalers(records: Mapping[str, ScalerRecord]) -> dict[str, WrittenArtifact]:
    written: dict[str, WrittenArtifact] = {}
    for asset, record in records.items():
        artifact = write_content_addressed(
            OUTPUT_ROOT / "scalers" / PILOT_FOLD / asset,
            "patchtst_candidate_scaler",
            ".json",
            record.payload,
        )
        if artifact.sha256 != record.identifier:
            raise Phase2HBExecutionError("Scaler artifact identifier changed during write.")
        written[asset] = artifact
    if len(written) != EXPECTED_SCALERS:
        raise Phase2HBExecutionError("Exactly four pilot scalers must be written.")
    return written


def _write_checkpoint(result: CandidateFitResult) -> WrittenArtifact:
    audit = result.audit
    return write_content_addressed(
        OUTPUT_ROOT
        / "models"
        / PILOT_FOLD
        / str(audit["model_config_id"])
        / str(audit["asset"])
        / str(audit["seed"]),
        "patchtst_candidate_best_state",
        ".pt",
        result.checkpoint_payload,
    )


def _coverage_with_authoritative_keys(
    predictions: pd.DataFrame, authoritative: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    audit = validation_coverage_audit(predictions)
    authoritative_records: list[dict[str, Any]] = []
    for asset in TARGET_ASSETS:
        expected = authoritative[asset][["asset", "origin_date", "target_date"]].sort_values(
            ["asset", "origin_date", "target_date"], kind="mergesort", ignore_index=True
        )
        expected_keys = pd.MultiIndex.from_frame(expected)
        expected_hash = sha256_bytes(canonical_csv_bytes(expected))
        for candidate in FROZEN_CANDIDATES:
            for seed in PILOT_SEEDS:
                part = predictions.loc[
                    (predictions["asset"] == asset)
                    & (predictions["model_config_id"] == candidate.config_id)
                    & (predictions["seed"] == seed),
                    ["asset", "origin_date", "target_date"],
                ].sort_values(["asset", "origin_date", "target_date"], kind="mergesort", ignore_index=True)
                if not pd.MultiIndex.from_frame(part).equals(expected_keys):
                    raise Phase2HBExecutionError(
                        f"Saved slice omitted or added an authoritative D1 key for {asset}."
                    )
        authoritative_records.append(
            {
                "asset": asset,
                "authoritative_validation_rows": len(expected),
                "authoritative_key_sha256": expected_hash,
                "all_four_context_seed_slices_exact": True,
            }
        )
    audit["authoritative_validation"] = authoritative_records
    audit["every_slice_equals_full_eligible_validation_set"] = True
    return audit


def _artifact_record(artifact: WrittenArtifact) -> dict[str, Any]:
    value: dict[str, Any] = {"path": _relative(artifact.path), "sha256": artifact.sha256}
    if artifact.rows is not None:
        value["rows"] = artifact.rows
    return value


def execute_pilot(
    *,
    command_started: float,
    contract: PatchTSTContract,
    environment_manifest: Mapping[str, Any],
    environment_artifact: WrittenArtifact,
    dry_run: Mapping[str, Any],
    dry_run_artifact: WrittenArtifact,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute exactly 16 candidate fits and the single separate P5 replay."""

    _assert_prefit_artifacts_only(environment_artifact, dry_run_artifact)
    plan = execution_plan()
    candidate_results: list[CandidateFitResult] = []
    for index, item in enumerate(plan["candidate_fits"], start=1):
        candidate = next(
            value
            for value in FROZEN_CANDIDATES
            if value.config_id == item["model_config_id"]
        )
        asset = str(item["asset"])
        seed = int(item["seed"])
        result = fit_candidate(
            prepared["sequences"][(candidate.config_id, asset, "train")],
            prepared["sequences"][(candidate.config_id, asset, "validation")],
            run_id=RUN_ID,
            candidate=candidate,
            asset=asset,
            seed=seed,
            scaler_identifier=prepared["scalers"][asset].identifier,
            processed_data_sha256=PROCESSED_SHA256,
            patchtst_config_sha256=PATCHTST_SEARCH_SHA256,
        )
        candidate_results.append(result)
        print(
            f"Completed PatchTST pilot candidate fit {index}/{EXPECTED_CANDIDATE_FITS}: "
            f"{candidate.config_id} {asset} seed={seed} "
            f"best_epoch={result.audit['best_epoch']}",
            flush=True,
        )

    replay_candidate = FROZEN_CANDIDATES[0]
    replay = fit_candidate(
        prepared["sequences"][(replay_candidate.config_id, "AAPL", "train")],
        prepared["sequences"][(replay_candidate.config_id, "AAPL", "validation")],
        run_id=RUN_ID,
        candidate=replay_candidate,
        asset="AAPL",
        seed=1729,
        scaler_identifier=prepared["scalers"]["AAPL"].identifier,
        processed_data_sha256=PROCESSED_SHA256,
        patchtst_config_sha256=PATCHTST_SEARCH_SHA256,
    )
    print("Completed P5 deterministic replay 1/1: PATCHTST_63 AAPL seed=1729", flush=True)

    audits, predictions = candidate_results_frames(candidate_results)
    validate_complete_candidate_grid(audits, predictions)
    coverage = _coverage_with_authoritative_keys(
        predictions, prepared["authoritative_validation"]
    )
    original = next(
        result
        for result in candidate_results
        if result.audit["model_config_id"] == "PATCHTST_63"
        and result.audit["asset"] == "AAPL"
        and int(result.audit["seed"]) == 1729
    )
    replay_verification = verify_deterministic_replay(original, replay)

    epoch_history = pd.concat(
        [result.epoch_history for result in candidate_results], ignore_index=True
    )
    finite_epoch_values = bool(
        np.isfinite(
            epoch_history[
                [
                    "train_row_weighted_l1",
                    "validation_mae",
                    "validation_rmse",
                    "validation_directional_accuracy",
                ]
            ].to_numpy(dtype=np.float64)
        ).all()
    )
    epoch_count_matches = all(
        len(
            epoch_history.loc[
                (epoch_history["model_config_id"] == row.model_config_id)
                & (epoch_history["asset"] == row.asset)
                & (epoch_history["seed"] == row.seed)
            ]
        )
        == int(row.epochs_run)
        for row in audits.itertuples(index=False)
    )
    numerical_validity = {
        "schema_version": 1,
        "status": "PASS"
        if finite_epoch_values
        and epoch_count_matches
        and bool(
            (epoch_history["loss_prediction_gradient_parameter_checks"] == "PASS").all()
        )
        and bool(np.isfinite(predictions["predicted_log_return"].to_numpy(float)).all())
        and bool(
            np.isfinite(
                audits[
                    [
                        "validation_mae",
                        "validation_rmse",
                        "validation_directional_accuracy",
                        "actual_positive_rate",
                    ]
                ].to_numpy(float)
            ).all()
        )
        else "FAIL",
        "candidate_fits": len(audits),
        "epoch_records": len(epoch_history),
        "all_epoch_losses_predictions_metrics_finite": finite_epoch_values,
        "all_gradient_and_parameter_checks_passed": bool(
            (epoch_history["loss_prediction_gradient_parameter_checks"] == "PASS").all()
        ),
        "epoch_history_matches_fit_epochs_run": epoch_count_matches,
        "all_final_predictions_finite": bool(
            np.isfinite(predictions["predicted_log_return"].to_numpy(float)).all()
        ),
        "all_required_fit_metrics_finite": bool(
            np.isfinite(
                audits[
                    [
                        "validation_mae",
                        "validation_rmse",
                        "validation_directional_accuracy",
                        "actual_positive_rate",
                    ]
                ].to_numpy(float)
            ).all()
        ),
    }

    scaler_artifacts = _write_scalers(prepared["scalers"])
    checkpoint_artifacts = [_write_checkpoint(result) for result in candidate_results]
    if len(checkpoint_artifacts) != EXPECTED_CANDIDATE_FITS:
        raise Phase2HBExecutionError("Exactly 16 candidate checkpoints must be written.")
    prediction_artifact = write_content_addressed(
        OUTPUT_ROOT / "predictions",
        "patchtst_pilot_validation_predictions",
        ".csv",
        canonical_csv_bytes(predictions),
        rows=len(predictions),
    )
    metrics_artifact = write_content_addressed(
        OUTPUT_ROOT / "metrics",
        "patchtst_pilot_fit_metrics",
        ".csv",
        canonical_csv_bytes(audits.loc[:, FIT_AUDIT_COLUMNS]),
        rows=len(audits),
    )
    epoch_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_pilot_epoch_numerical_history",
        ".csv",
        canonical_csv_bytes(epoch_history),
        rows=len(epoch_history),
    )
    numerical_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_pilot_numerical_validity",
        ".json",
        canonical_json_bytes(numerical_validity),
    )
    coverage_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_pilot_validation_coverage",
        ".json",
        canonical_json_bytes(coverage),
    )
    replay_prediction_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_p5_replay_predictions",
        ".csv",
        canonical_csv_bytes(replay.predictions),
        rows=len(replay.predictions),
    )
    replay_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_p5_deterministic_replay",
        ".json",
        canonical_json_bytes(replay_verification),
    )

    frozen_after = verify_frozen_identities(contract)
    source_matches_environment = (
        source_tree_identity()["sha256"]
        == environment_manifest["source_tree_identity"]["sha256"]
    )
    integrity = {
        "schema_version": 1,
        "status": "PASS" if frozen_after["status"] == "PASS" and source_matches_environment else "FAIL",
        "frozen_inputs": frozen_after,
        "source_matches_prefit_environment_manifest": source_matches_environment,
        "patchtst_preregistration_unchanged": True,
        "patchtst_search_unchanged": True,
        "historical_artifact_trees_unchanged": True,
        "d1_test_predictions_generated": False,
        "d2_d5_model_performance_accessed": False,
        "f1_2025_performance_evaluated": False,
        "regime_analysis_performed": False,
    }
    integrity_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_pilot_integrity",
        ".json",
        canonical_json_bytes(integrity),
    )

    observed_before_finalization = perf_counter() - command_started
    conservative_total = observed_before_finalization + FINALIZATION_ALLOWANCE_SECONDS
    resource = compute_resource_feasibility(conservative_total)
    resource.update(
        {
            "timer_started_at_non_dry_run_command_entry": True,
            "observed_seconds_through_training_replay_verification_and_primary_writes": observed_before_finalization,
            "finalization_allowance_seconds": FINALIZATION_ALLOWANCE_SECONDS,
            "recorded_total_is_conservative_upper_bound": True,
            "measured_scope_includes_candidate_fits": 16,
            "measured_scope_includes_replay": 1,
            "measured_scope_includes_runner_verification_and_writes": True,
            "memory": _current_process_peak_memory(),
        }
    )
    resource_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_pilot_resource_feasibility",
        ".json",
        canonical_json_bytes(resource),
    )

    reloaded_audits = pd.read_csv(metrics_artifact.path)
    reloaded_predictions = pd.read_csv(prediction_artifact.path)
    validate_complete_candidate_grid(reloaded_audits, reloaded_predictions)
    reloaded_numerical = json.loads(numerical_artifact.path.read_text(encoding="utf-8"))
    reloaded_coverage = json.loads(coverage_artifact.path.read_text(encoding="utf-8"))
    reloaded_replay = json.loads(replay_artifact.path.read_text(encoding="utf-8"))
    gate = build_gate_decision(
        integrity_passed=integrity["status"] == "PASS",
        grid_passed=len(reloaded_audits) == EXPECTED_CANDIDATE_FITS,
        numerical_validity_passed=reloaded_numerical["status"] == "PASS",
        coverage_passed=reloaded_coverage["status"] == "PASS"
        and reloaded_coverage["every_slice_equals_full_eligible_validation_set"] is True,
        replay_passed=reloaded_replay["status"] == "PASS",
        resource=resource,
    )
    gate_artifact = write_content_addressed(
        OUTPUT_ROOT / "gate",
        "patchtst_pilot_gate_decision",
        ".json",
        canonical_json_bytes(gate),
    )

    independent = independently_verify_saved_pilot(
        prediction_path=prediction_artifact.path,
        metrics_path=metrics_artifact.path,
        gate_path=gate_artifact.path,
    )
    independent_artifact = write_content_addressed(
        OUTPUT_ROOT / "verification",
        "patchtst_pilot_independent_saved_file_verification",
        ".json",
        canonical_json_bytes(independent),
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "phase2hb_patchtst_pilot_run_manifest",
        "run_id": RUN_ID,
        "created_timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": "completed",
        "decision": gate["decision"],
        "source_tree_identity": source_tree_identity(),
        "environment": _artifact_record(environment_artifact),
        "dry_run": _artifact_record(dry_run_artifact),
        "frozen_inputs": frozen_after,
        "contract": {
            "model": "PatchTST-style channel-independent patch Transformer",
            "fold": PILOT_FOLD,
            "partitions": ["train", "validation"],
            "features": list(FROZEN_FEATURE_COLUMNS),
            "candidates": [item.__dict__ for item in FROZEN_CANDIDATES],
            "pilot_seeds": list(PILOT_SEEDS),
            "candidate_fits": EXPECTED_CANDIDATE_FITS,
            "replay_executions": EXPECTED_REPLAY_EXECUTIONS,
            "refits": 0,
            "test_predictions": 0,
            "model_selection_performed": False,
            "performance_gate_used": False,
        },
        "counts": {
            "candidate_scalers": len(scaler_artifacts),
            "candidate_fits": len(audits),
            "candidate_checkpoints": len(checkpoint_artifacts),
            "candidate_validation_prediction_rows": len(predictions),
            "p5_replay_executions": 1,
            "p5_replay_prediction_rows": len(replay.predictions),
            "total_training_executions": EXPECTED_TOTAL_TRAINING_EXECUTIONS,
        },
        "artifacts": {
            "scalers": {key: _artifact_record(value) for key, value in scaler_artifacts.items()},
            "candidate_models": [_artifact_record(value) for value in checkpoint_artifacts],
            "predictions": _artifact_record(prediction_artifact),
            "fit_metrics": _artifact_record(metrics_artifact),
            "epoch_history": _artifact_record(epoch_artifact),
            "numerical_validity": _artifact_record(numerical_artifact),
            "validation_coverage": _artifact_record(coverage_artifact),
            "p5_replay_predictions": _artifact_record(replay_prediction_artifact),
            "p5_replay_verification": _artifact_record(replay_artifact),
            "integrity": _artifact_record(integrity_artifact),
            "resource_feasibility": _artifact_record(resource_artifact),
            "gate": _artifact_record(gate_artifact),
            "independent_saved_file_verification": _artifact_record(independent_artifact),
        },
        "gate": gate,
        "resource": resource,
        "protected_boundaries": {
            "d1_validation_only": True,
            "d1_test_predictions_generated": False,
            "d2_d5_model_performance_accessed": False,
            "f1_2025_performance_evaluated": False,
            "regime_analysis_performed": False,
            "full_benchmark_executed": False,
        },
        "specification_deviations": [],
    }
    manifest_artifact = write_content_addressed(
        OUTPUT_ROOT,
        "run_manifest",
        ".json",
        canonical_json_bytes(manifest),
    )
    actual_elapsed = perf_counter() - command_started
    if actual_elapsed > conservative_total:
        raise Phase2HBExecutionError(
            "Finalization exceeded the conservative recorded pilot-time allowance."
        )
    return {
        "manifest": manifest,
        "manifest_artifact": manifest_artifact,
        "gate": gate,
        "gate_artifact": gate_artifact,
        "metrics": audits,
        "predictions": predictions,
        "resource": resource,
        "actual_elapsed_before_process_exit": actual_elapsed,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 2H-B D1 train/validation PatchTST pilot."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Capture environment and verify the exact pilot without fitting a model.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    command_started = perf_counter()
    contract = load_patchtst_contract(PATCHTST_SEARCH_PATH)
    frozen = verify_frozen_identities(contract)
    environment_manifest, environment_artifact = capture_environment_manifest(contract)
    dry_run, prepared = perform_dry_run(
        contract=contract,
        environment_manifest=environment_manifest,
        environment_artifact=environment_artifact,
        frozen=frozen,
    )
    if args.dry_run:
        dry_run_artifact = _write_dry_run_receipt(dry_run)
        output = dict(dry_run)
        output["dry_run_receipt"] = _artifact_record(dry_run_artifact)
        print(json.dumps(output, indent=2, default=str), flush=True)
        return 0
    dry_run_artifact = _require_prior_dry_run(dry_run)
    result = execute_pilot(
        command_started=command_started,
        contract=contract,
        environment_manifest=environment_manifest,
        environment_artifact=environment_artifact,
        dry_run=dry_run,
        dry_run_artifact=dry_run_artifact,
        prepared=prepared,
    )
    print(f"PatchTST pilot manifest: {_relative(result['manifest_artifact'].path)}")
    print(f"PatchTST pilot gate: {_relative(result['gate_artifact'].path)}")
    print(f"Candidate fits completed: {len(result['metrics'])}/16")
    print("P5 deterministic replay completed: Yes")
    print(f"Pilot decision: {result['gate']['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
