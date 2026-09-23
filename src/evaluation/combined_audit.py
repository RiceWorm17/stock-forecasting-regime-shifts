"""Phase 2G read-only audit of frozen D1--D5 benchmark predictions.

This module deliberately has no dependency on model implementations or training
evaluation modules.  Its only permitted inputs are frozen prediction/metric
artifacts; its only permitted writes are new, content-addressed files below
``results/combined``.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMBINED_ROOT = REPOSITORY_ROOT / "results" / "combined"

SPEC_VERSION = "1.1"
FOLDS = ("D1", "D2", "D3", "D4", "D5")
ASSETS = ("AAPL", "MSFT", "GOOGL", "NVDA")
LSTM_SEEDS = (1729, 2718, 31415)
KEY_COLUMNS = ("fold", "asset", "origin_date", "target_date")
REGIME_COLUMNS = ("trend_regime", "volatility_regime", "transition_regime")
PREDICTION_COLUMNS = (
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
    *REGIME_COLUMNS,
)

NUMERIC_TOLERANCE = 1e-12
ACTUAL_RTOL = 1e-12
ACTUAL_ATOL = 1e-15
ALLOWED_GATE_STATUSES = frozenset({"PASS", "FAIL", "CONDITIONAL PASS"})
ALLOWED_GATE_DECISIONS = frozenset(
    {
        "ADMIT PATCHTST TO A PREREGISTERED PILOT",
        "DO NOT ADMIT PATCHTST",
    }
)

BASELINE_PREDICTION_SHA256 = (
    "e09e4909e9af7b7836c5e0b2c8d7b320d2848d5da9b33b8c9492421e0a108bb4"
)
LIGHTGBM_PREDICTION_SHA256 = (
    "36445836777cba815bc240c0c7229838cdc2177e94f50bac7905b8e2a9ace0c7"
)
LSTM_PREDICTION_SHA256 = (
    "f624bc00c41a430c6e79188fe8b10a1bc03714971719789d5162446c70c7df60"
)
PREREGISTRATION_SHA256 = (
    "be04b9ce8457c5b26a8042f548eccce8ddb4b5d1b86a1404a43ef1bb1a52d613"
)
EXPERIMENT_SPEC_SHA256 = (
    "4a29b96a7c826634d2dede63220c378737c0e545a7e733491a0a7002d69fc5b9"
)

FROZEN_FILES: Mapping[str, tuple[str, str, str]] = {
    "baseline_predictions": (
        "results/baselines/predictions/"
        f"baseline_predictions_{BASELINE_PREDICTION_SHA256}.csv",
        BASELINE_PREDICTION_SHA256,
        "PHASE_2C_REPORT.md",
    ),
    "baseline_asset_metrics": (
        "results/baselines/metrics/"
        "baseline_metrics_by_asset_"
        "ac1a32f81a6cf597e186f317ca2983ac15c0cf45c75c2c0968a4dce85e6a6a16.csv",
        "ac1a32f81a6cf597e186f317ca2983ac15c0cf45c75c2c0968a4dce85e6a6a16",
        "PHASE_2C_REPORT.md",
    ),
    "baseline_macro_metrics": (
        "results/baselines/metrics/"
        "baseline_metrics_macro_"
        "27bb8779cf550b7cf288e098f7551eac43100879afe09786265ef70d69be65c8.csv",
        "27bb8779cf550b7cf288e098f7551eac43100879afe09786265ef70d69be65c8",
        "PHASE_2C_REPORT.md",
    ),
    "baseline_manifest": (
        "results/baselines/"
        "run_manifest_9927474b9a3ef014ad4e016c48a7f8cda461a0ccb3e0daaab6f124135ddc32d0.json",
        "9927474b9a3ef014ad4e016c48a7f8cda461a0ccb3e0daaab6f124135ddc32d0",
        "PHASE_2C_REPORT.md",
    ),
    "lightgbm_predictions": (
        "results/lightgbm/predictions/"
        f"lightgbm_development_predictions_{LIGHTGBM_PREDICTION_SHA256}.csv",
        LIGHTGBM_PREDICTION_SHA256,
        "PHASE_2E_REPORT.md",
    ),
    "lightgbm_asset_metrics": (
        "results/lightgbm/metrics/"
        "lightgbm_metrics_by_asset_"
        "89aa75ff8d6833eb9f0089c94126307fd94bdf93899375bcb6b5616c86e5a5e5.csv",
        "89aa75ff8d6833eb9f0089c94126307fd94bdf93899375bcb6b5616c86e5a5e5",
        "PHASE_2E_REPORT.md",
    ),
    "lightgbm_macro_metrics": (
        "results/lightgbm/metrics/"
        "lightgbm_metrics_macro_"
        "ce1ad407b42129b78f2c654f9c797aecb22ea916f6ee3e9ba229efcd386a879e.csv",
        "ce1ad407b42129b78f2c654f9c797aecb22ea916f6ee3e9ba229efcd386a879e",
        "PHASE_2E_REPORT.md",
    ),
    "lightgbm_manifest": (
        "results/lightgbm/"
        "run_manifest_6b90c0907a42933b3d184bc87a6bc720dbf846a9d751e475c2657539e815bf75.json",
        "6b90c0907a42933b3d184bc87a6bc720dbf846a9d751e475c2657539e815bf75",
        "PHASE_2E_REPORT.md",
    ),
    "lstm_predictions": (
        "results/lstm/predictions/"
        f"lstm_development_predictions_{LSTM_PREDICTION_SHA256}.csv",
        LSTM_PREDICTION_SHA256,
        "PHASE_2F_REPORT.md",
    ),
    "lstm_seed_metrics": (
        "results/lstm/metrics/"
        "lstm_metrics_by_asset_seed_"
        "666a6caeae13744df79919e2f30466c9c5a025c5eb2966e785f90c0469159ede.csv",
        "666a6caeae13744df79919e2f30466c9c5a025c5eb2966e785f90c0469159ede",
        "PHASE_2F_REPORT.md",
    ),
    "lstm_asset_seed_summary": (
        "results/lstm/metrics/"
        "lstm_metrics_asset_seed_summary_"
        "e2d2cae825530d985d6bac671290f0a1ccd2a250043fb3617cfbfc3c4f47aa6a.csv",
        "e2d2cae825530d985d6bac671290f0a1ccd2a250043fb3617cfbfc3c4f47aa6a",
        "PHASE_2F_REPORT.md",
    ),
    "lstm_macro_by_seed": (
        "results/lstm/metrics/"
        "lstm_metrics_macro_by_seed_"
        "f833635ae0e159f5d797886a10aa9136ece637eaaa0091d854e592b000a9eb33.csv",
        "f833635ae0e159f5d797886a10aa9136ece637eaaa0091d854e592b000a9eb33",
        "PHASE_2F_REPORT.md",
    ),
    "lstm_macro_seed_summary": (
        "results/lstm/metrics/"
        "lstm_metrics_macro_seed_summary_"
        "d0ad5cc9b434e53e856f60ff03b37f6d184aa40b98b81da15540e19750148a17.csv",
        "d0ad5cc9b434e53e856f60ff03b37f6d184aa40b98b81da15540e19750148a17",
        "PHASE_2F_REPORT.md",
    ),
    "lstm_manifest": (
        "results/lstm/"
        "run_manifest_a657eaa1426e369ff68275c75bac0e4e8c4e91f33fefde8d7849c51eb85df283.json",
        "a657eaa1426e369ff68275c75bac0e4e8c4e91f33fefde8d7849c51eb85df283",
        "PHASE_2F_REPORT.md",
    ),
}

FROZEN_TREE_IDENTITIES: Mapping[str, Mapping[str, Any]] = {
    "baseline": {
        "path": "results/baselines",
        "file_count": 6,
        "sha256": "1f42bb6758b54251060983e1b7a9323861d94ac768ee5371571e77e5b46fc466",
    },
    "lightgbm": {
        "path": "results/lightgbm",
        "file_count": 113,
        "sha256": "6f1079692067a8ed49dcce3d3db52efd22e00883084575a3a804a6f942e9d970",
    },
    "lstm": {
        "path": "results/lstm",
        "file_count": 235,
        "sha256": "64d0866469259062e49652c17be364e34cde6bbe7fecc17dbb821489e6b66f61",
    },
}

PROVENANCE_COLUMNS = (
    "generation_timestamp_utc",
    "experiment_spec_version",
    "experiment_spec_sha256",
    "model_preregistration_path",
    "model_preregistration_sha256",
    "baseline_prediction_sha256",
    "lightgbm_prediction_sha256",
    "lstm_prediction_sha256",
    "source_tree_sha256",
    "source_revision",
)


class CombinedAuditError(RuntimeError):
    """Raised when Phase 2G finds an integrity or audit-contract violation."""


@dataclass(frozen=True)
class RecomputedMetrics:
    baseline_asset: pd.DataFrame
    baseline_macro: pd.DataFrame
    lightgbm_asset: pd.DataFrame
    lightgbm_macro: pd.DataFrame
    lstm_seed: pd.DataFrame
    lstm_asset_seed_summary: pd.DataFrame
    lstm_macro_by_seed: pd.DataFrame
    lstm_macro_seed_summary: pd.DataFrame


@dataclass(frozen=True)
class CombinedAuditResult:
    artifact_paths: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]
    integrity: Mapping[str, Any]
    key_audit: Mapping[str, Any]
    metric_differences: Mapping[str, float]
    gate_decision: Mapping[str, Any]
    summary: Mapping[str, Any]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(
        stream,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return stream.getvalue().encode("utf-8")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CombinedAuditError(f"Refusing to overwrite immutable artifact {path}.")
        return
    path.write_bytes(payload)


def _write_content_addressed_csv(frame: pd.DataFrame, stem: str) -> tuple[Path, str]:
    payload = canonical_csv_bytes(frame)
    digest = sha256_bytes(payload)
    path = COMBINED_ROOT / f"{stem}_{digest}.csv"
    _write_immutable(path, payload)
    return path, digest


def _write_content_addressed_json(
    value: Mapping[str, Any], stem: str
) -> tuple[Path, str]:
    payload = canonical_json_bytes(value)
    digest = sha256_bytes(payload)
    path = COMBINED_ROOT / f"{stem}_{digest}.json"
    _write_immutable(path, payload)
    return path, digest


def tree_identity(relative_root: str) -> dict[str, Any]:
    root = (REPOSITORY_ROOT / relative_root).resolve()
    if not root.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise CombinedAuditError(f"Tree path escaped repository: {relative_root}.")
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
        "sha256": sha256_bytes("\n".join(records).encode("utf-8")),
    }


def source_tree_identity() -> dict[str, Any]:
    root = REPOSITORY_ROOT / "src"
    files = sorted(
        (path for path in root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix(),
    )
    digest = hashlib.sha256()
    per_file: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        file_digest = file_sha256(path)
        per_file[relative] = file_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "method": "sha256_of_sorted_relative_path_nul_file_sha256_lf_records",
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": per_file,
        "git_revision": None,
        "git_status": "not_available_repository_has_no_git_metadata",
    }


def _collect_manifest_artifact_records(value: Any) -> set[tuple[str, str]]:
    records: set[tuple[str, str]] = set()
    if isinstance(value, Mapping):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            records.add((value["path"].replace("\\", "/"), value["sha256"]))
        for child in value.values():
            records.update(_collect_manifest_artifact_records(child))
    elif isinstance(value, list):
        for child in value:
            records.update(_collect_manifest_artifact_records(child))
    return records


def verify_frozen_integrity() -> dict[str, Any]:
    """Hash-gate all authoritative inputs, manifests, reports, and frozen trees."""

    actual_files: dict[str, dict[str, Any]] = {}
    reports: dict[str, str] = {}
    for name, (relative, expected, report_relative) in FROZEN_FILES.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise CombinedAuditError(f"Missing frozen artifact: {relative}.")
        actual = file_sha256(path)
        if actual != expected:
            raise CombinedAuditError(
                f"Frozen artifact hash mismatch for {relative}: {actual} != {expected}."
            )
        report_text = reports.setdefault(
            report_relative,
            (REPOSITORY_ROOT / report_relative).read_text(encoding="utf-8"),
        )
        if expected not in report_text:
            raise CombinedAuditError(
                f"{report_relative} does not record the frozen digest for {relative}."
            )
        actual_files[name] = {
            "path": relative,
            "sha256": actual,
            "phase_report": report_relative,
            "report_reference_verified": True,
        }

    for manifest_name, prefix in (
        ("baseline_manifest", "baseline_"),
        ("lightgbm_manifest", "lightgbm_"),
        ("lstm_manifest", "lstm_"),
    ):
        manifest_path = REPOSITORY_ROOT / FROZEN_FILES[manifest_name][0]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = _collect_manifest_artifact_records(manifest)
        for name, (relative, expected, _) in FROZEN_FILES.items():
            if name == manifest_name or not name.startswith(prefix):
                continue
            matched = any(
                record_hash == expected
                and (
                    record_path == relative
                    or relative.endswith(record_path)
                    or record_path.endswith(relative)
                )
                for record_path, record_hash in records
            )
            if not matched:
                raise CombinedAuditError(
                    f"{manifest_path.name} does not bind {relative} to {expected}."
                )

    for relative, expected in (
        ("docs/EXPERIMENT_SPEC.md", EXPERIMENT_SPEC_SHA256),
        ("docs/MODEL_PREREGISTRATION.md", PREREGISTRATION_SHA256),
    ):
        actual = file_sha256(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise CombinedAuditError(
                f"Authoritative contract changed: {relative} is {actual}, expected {expected}."
            )

    trees: dict[str, dict[str, Any]] = {}
    for name, expected in FROZEN_TREE_IDENTITIES.items():
        actual = tree_identity(str(expected["path"]))
        if actual["file_count"] != expected["file_count"] or actual["sha256"] != expected["sha256"]:
            raise CombinedAuditError(
                f"Frozen {name} tree changed: {actual} != {dict(expected)}."
            )
        trees[name] = actual

    return {
        "status": "PASS",
        "files": actual_files,
        "trees": trees,
        "experiment_spec": {
            "path": "docs/EXPERIMENT_SPEC.md",
            "version": SPEC_VERSION,
            "sha256": EXPERIMENT_SPEC_SHA256,
        },
        "model_preregistration": {
            "path": "docs/MODEL_PREREGISTRATION.md",
            "sha256": PREREGISTRATION_SHA256,
        },
    }


def load_prediction_artifact(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"fold": "string", "asset": "string"})
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise CombinedAuditError(
            f"Prediction columns in {path} differ from the frozen 18-column contract."
        )
    for column in ("origin_date", "target_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    for column in ("actual_log_return", "predicted_log_return"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    for column in ("actual_direction", "predicted_direction"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int8")
    return frame


def validate_prediction_scope(
    frame: pd.DataFrame,
    *,
    family: str,
    expected_rows: int,
    expected_regime_placeholder: str,
) -> None:
    if len(frame) != expected_rows:
        raise CombinedAuditError(
            f"{family} prediction rows are {len(frame)}, expected {expected_rows}."
        )
    if set(frame["fold"].astype(str)) != set(FOLDS):
        raise CombinedAuditError(f"{family} predictions must contain only and all D1--D5.")
    if set(frame["asset"].astype(str)) != set(ASSETS):
        raise CombinedAuditError(f"{family} predictions must contain exactly four target assets.")
    if set(frame["partition"].astype(str)) != {"test"}:
        raise CombinedAuditError(f"{family} predictions must be partition=test only.")
    years = frame["target_date"].dt.year
    if (years >= 2025).any() or set(years) != {2020, 2021, 2022, 2023, 2024}:
        raise CombinedAuditError(f"{family} predictions escaped the D1--D5 2020--2024 scope.")
    expected_year_by_fold = {fold: 2020 + index for index, fold in enumerate(FOLDS)}
    actual_year_by_fold = frame.groupby("fold", observed=True)["target_date"].apply(
        lambda values: set(values.dt.year)
    )
    for fold, expected_year in expected_year_by_fold.items():
        if actual_year_by_fold.loc[fold] != {expected_year}:
            raise CombinedAuditError(f"{family} {fold} target-year mapping is invalid.")
    if not np.isfinite(frame[["actual_log_return", "predicted_log_return"]].to_numpy()).all():
        raise CombinedAuditError(f"{family} predictions contain non-finite regression values.")
    if not frame["actual_direction"].isin([0, 1]).all() or not frame[
        "predicted_direction"
    ].isin([0, 1]).all():
        raise CombinedAuditError(f"{family} predictions contain non-binary directions.")
    expected_actual_direction = (frame["actual_log_return"] > 0).astype("int8")
    if not np.array_equal(
        frame["actual_direction"].to_numpy(dtype="int8"),
        expected_actual_direction.to_numpy(dtype="int8"),
    ):
        raise CombinedAuditError(f"{family} actual_direction disagrees with actual_log_return.")
    for column in REGIME_COLUMNS:
        if set(frame[column].astype(str)) != {expected_regime_placeholder}:
            raise CombinedAuditError(
                f"{family} {column} changed from sentinel {expected_regime_placeholder}."
            )

    duplicate_columns = list(KEY_COLUMNS)
    if family == "lstm":
        duplicate_columns.append("seed")
    if frame.duplicated(duplicate_columns, keep=False).any():
        raise CombinedAuditError(f"{family} predictions contain duplicate canonical keys.")

    if family == "baseline":
        if set(frame["seed"].astype(str)) != {"not_applicable"}:
            raise CombinedAuditError("Baseline seed identity changed.")
        if not np.array_equal(frame["predicted_log_return"].to_numpy(), np.zeros(len(frame))):
            raise CombinedAuditError("Baseline regression predictions are no longer zero-return.")
    else:
        numeric_seed = pd.to_numeric(frame["seed"], errors="raise").astype(int)
        expected_seeds = {1729} if family == "lightgbm" else set(LSTM_SEEDS)
        if set(numeric_seed) != expected_seeds:
            raise CombinedAuditError(f"{family} seeds do not match {sorted(expected_seeds)}.")
        expected_predicted_direction = (frame["predicted_log_return"] > 0).astype("int8")
        if not np.array_equal(
            frame["predicted_direction"].to_numpy(dtype="int8"),
            expected_predicted_direction.to_numpy(dtype="int8"),
        ):
            raise CombinedAuditError(
                f"{family} predicted_direction is not sign-derived from its saved forecast."
            )


def _sorted_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, KEY_COLUMNS].sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def _assert_key_equality(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    left = _sorted_key_frame(reference)
    right = _sorted_key_frame(candidate)
    if len(left) != len(right) or not left.equals(right):
        raise CombinedAuditError(f"Canonical key mismatch for {label}; intersections are prohibited.")


def _actual_agreement(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> float:
    left = reference.loc[:, [*KEY_COLUMNS, "actual_log_return", "actual_direction"]]
    right = candidate.loc[:, [*KEY_COLUMNS, "actual_log_return", "actual_direction"]]
    merged = left.merge(
        right,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
        indicator=True,
    )
    if set(merged["_merge"].astype(str)) != {"both"}:
        raise CombinedAuditError(f"Actual-value comparison keys differ for {label}.")
    reference_values = merged["actual_log_return_reference"].to_numpy(dtype="float64")
    candidate_values = merged["actual_log_return_candidate"].to_numpy(dtype="float64")
    differences = np.abs(reference_values - candidate_values)
    if not np.allclose(reference_values, candidate_values, rtol=ACTUAL_RTOL, atol=ACTUAL_ATOL):
        raise CombinedAuditError(f"actual_log_return mismatch for {label}.")
    if not np.array_equal(
        merged["actual_direction_reference"].to_numpy(),
        merged["actual_direction_candidate"].to_numpy(),
    ):
        raise CombinedAuditError(f"actual_direction mismatch for {label}.")
    return float(differences.max(initial=0.0))


def audit_canonical_keys(
    baseline: pd.DataFrame, lightgbm: pd.DataFrame, lstm: pd.DataFrame
) -> dict[str, Any]:
    _assert_key_equality(baseline, lightgbm, "LightGBM")
    max_actual_difference: dict[str, float] = {
        "lightgbm": _actual_agreement(baseline, lightgbm, "LightGBM")
    }
    seed_counts: dict[str, int] = {}
    for seed in LSTM_SEEDS:
        seed_frame = lstm.loc[pd.to_numeric(lstm["seed"]) == seed].copy()
        _assert_key_equality(baseline, seed_frame, f"LSTM seed {seed}")
        max_actual_difference[str(seed)] = _actual_agreement(
            baseline, seed_frame, f"LSTM seed {seed}"
        )
        seed_counts[str(seed)] = len(seed_frame)
    return {
        "status": "PASS",
        "key_columns": list(KEY_COLUMNS),
        "unique_keys": len(_sorted_key_frame(baseline)),
        "baseline_rows": len(baseline),
        "lightgbm_rows": len(lightgbm),
        "lstm_rows": len(lstm),
        "lstm_seed_rows": seed_counts,
        "no_intersection_used": True,
        "max_abs_actual_log_return_difference": max_actual_difference,
        "actual_direction_exact_match": True,
    }


def compute_baseline_asset_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_id, model, fold, asset), frame in predictions.groupby(
        ["run_id", "model", "fold", "asset"], sort=True, observed=True
    ):
        actual = frame["actual_log_return"].to_numpy(dtype="float64")
        predicted = frame["predicted_log_return"].to_numpy(dtype="float64")
        actual_direction = frame["actual_direction"].to_numpy(dtype="int8")
        predicted_direction = frame["predicted_direction"].to_numpy(dtype="int8")
        n = len(frame)
        positive_count = int(actual_direction.sum())
        errors = predicted - actual
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "fold": fold,
                "asset": asset,
                "n": n,
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "directional_accuracy": float(np.mean(predicted_direction == actual_direction)),
                "actual_positive_count": positive_count,
                "actual_nonpositive_count": n - positive_count,
                "actual_positive_rate": positive_count / n,
                "actual_nonpositive_rate": (n - positive_count) / n,
                "skill_score_vs_self": 0.0,
                "skill_interpretation": "zero_by_definition_not_meaningful",
            }
        )
    return pd.DataFrame(rows).sort_values(["fold", "asset"], kind="mergesort").reset_index(
        drop=True
    )


def compute_baseline_macro_metrics(asset_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_id, model, fold), frame in asset_metrics.groupby(
        ["run_id", "model", "fold"], sort=True, observed=True
    ):
        if set(frame["asset"]) != set(ASSETS) or len(frame) != len(ASSETS):
            raise CombinedAuditError(f"Baseline {fold} lacks one row per target asset.")
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "fold": fold,
                "n_assets": len(frame),
                "n_observations": int(frame["n"].sum()),
                "mae": float(frame["mae"].mean()),
                "rmse": float(frame["rmse"].mean()),
                "directional_accuracy": float(frame["directional_accuracy"].mean()),
                "actual_positive_rate": float(frame["actual_positive_rate"].mean()),
                "actual_nonpositive_rate": float(frame["actual_nonpositive_rate"].mean()),
                "macro_weighting": "equal_weight_across_assets",
                "skill_score_vs_self": 0.0,
                "skill_interpretation": "zero_by_definition_not_meaningful",
            }
        )
    return pd.DataFrame(rows).sort_values("fold", kind="mergesort").reset_index(drop=True)


def _join_matched_baseline(
    predictions: pd.DataFrame, baseline: pd.DataFrame, *, family: str
) -> pd.DataFrame:
    baseline_columns = [
        *KEY_COLUMNS,
        "actual_log_return",
        "actual_direction",
        "predicted_log_return",
        "predicted_direction",
    ]
    merged = predictions.merge(
        baseline.loc[:, baseline_columns],
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one" if family == "lstm" else "one_to_one",
        suffixes=("", "_baseline"),
    )
    if merged["actual_log_return_baseline"].isna().any():
        raise CombinedAuditError(f"{family} has unmatched baseline observations.")
    if not np.allclose(
        merged["actual_log_return"],
        merged["actual_log_return_baseline"],
        rtol=ACTUAL_RTOL,
        atol=ACTUAL_ATOL,
    ):
        raise CombinedAuditError(f"{family} actual returns differ from baseline.")
    if not np.array_equal(merged["actual_direction"], merged["actual_direction_baseline"]):
        raise CombinedAuditError(f"{family} actual directions differ from baseline.")
    return merged


def compute_learned_asset_metrics(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    family: str,
) -> pd.DataFrame:
    if family not in {"lightgbm", "lstm"}:
        raise ValueError("family must be lightgbm or lstm")
    merged = _join_matched_baseline(predictions, baseline, family=family)
    group_columns = ["run_id", "fold", "asset", "seed", "model_config_id"]
    rows: list[dict[str, Any]] = []
    for key, frame in merged.groupby(group_columns, sort=True, observed=True):
        run_id, fold, asset, seed, model_config_id = key
        actual = frame["actual_log_return"].to_numpy(dtype="float64")
        predicted = frame["predicted_log_return"].to_numpy(dtype="float64")
        actual_direction = frame["actual_direction"].to_numpy(dtype="int8")
        predicted_direction = frame["predicted_direction"].to_numpy(dtype="int8")
        baseline_predicted = frame["predicted_log_return_baseline"].to_numpy(dtype="float64")
        baseline_direction = frame["predicted_direction_baseline"].to_numpy(dtype="int8")
        mae = float(np.mean(np.abs(predicted - actual)))
        baseline_mae = float(np.mean(np.abs(baseline_predicted - actual)))
        da = float(np.mean(predicted_direction == actual_direction))
        baseline_da = float(np.mean(baseline_direction == actual_direction))
        rows.append(
            {
                "run_id": run_id,
                "fold": fold,
                "asset": asset,
                "seed": int(seed),
                "model_config_id": model_config_id,
                "n": len(frame),
                "mae": mae,
                "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
                "directional_accuracy": da,
                "actual_positive_rate": float(np.mean(actual_direction == 1)),
                "baseline_mae": baseline_mae,
                "mae_skill": float(1.0 - mae / baseline_mae),
                "baseline_directional_accuracy": baseline_da,
                "da_difference": da - baseline_da,
            }
        )
    result = pd.DataFrame(rows)
    if family == "lightgbm":
        prefix = predictions.iloc[0]
        result.insert(1, "spec_version", prefix["spec_version"])
        result.insert(2, "data_version", prefix["data_version"])
        result.insert(3, "model", "lightgbm")
        # The Phase 2E table includes explicit class counts; derive them here.
        result["actual_positive_count"] = (result["actual_positive_rate"] * result["n"]).round().astype(int)
        result["actual_nonpositive_count"] = result["n"] - result["actual_positive_count"]
        result["actual_nonpositive_rate"] = 1.0 - result["actual_positive_rate"]
        columns = [
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
        ]
        result = result.loc[:, columns]
    else:
        columns = [
            "run_id",
            "fold",
            "asset",
            "seed",
            "model_config_id",
            "n",
            "mae",
            "rmse",
            "directional_accuracy",
            "actual_positive_rate",
            "baseline_mae",
            "mae_skill",
            "baseline_directional_accuracy",
            "da_difference",
        ]
        result = result.loc[:, columns]
    return result.sort_values(["fold", "asset", "seed"], kind="mergesort").reset_index(drop=True)


_SUMMARY_METRICS = (
    "mae",
    "rmse",
    "directional_accuracy",
    "actual_positive_rate",
    "baseline_mae",
    "mae_skill",
    "baseline_directional_accuracy",
    "da_difference",
)


def compute_lstm_asset_seed_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_id, fold, asset, model_config_id), frame in seed_metrics.groupby(
        ["run_id", "fold", "asset", "model_config_id"], sort=True, observed=True
    ):
        if set(frame["seed"].astype(int)) != set(LSTM_SEEDS) or len(frame) != 3:
            raise CombinedAuditError(f"LSTM {fold}/{asset} lacks exactly three fixed seeds.")
        if frame["n"].nunique() != 1:
            raise CombinedAuditError(f"LSTM {fold}/{asset} seeds have unequal row counts.")
        row: dict[str, Any] = {
            "run_id": run_id,
            "fold": fold,
            "asset": asset,
            "model_config_id": model_config_id,
            "n_seeds": 3,
            "n_per_seed": int(frame["n"].iloc[0]),
        }
        for metric in _SUMMARY_METRICS:
            values = frame[metric].to_numpy(dtype="float64")
            row.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_std": float(values.std(ddof=0)),
                    f"{metric}_min": float(values.min()),
                    f"{metric}_max": float(values.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fold", "asset"], kind="mergesort").reset_index(
        drop=True
    )


def compute_lightgbm_macro_metrics(asset_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["run_id", "spec_version", "data_version", "model", "model_config_id", "seed", "fold"]
    for key, frame in asset_metrics.groupby(groups, sort=True, observed=True):
        if set(frame["asset"]) != set(ASSETS) or len(frame) != 4:
            raise CombinedAuditError("LightGBM macro metrics require four equal-weight assets.")
        row = dict(zip(groups, key, strict=True))
        row.update(
            {
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
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold", kind="mergesort").reset_index(drop=True)


def compute_lstm_macro_by_seed(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = ["run_id", "fold", "seed", "model_config_id"]
    for key, frame in seed_metrics.groupby(groups, sort=True, observed=True):
        if set(frame["asset"]) != set(ASSETS) or len(frame) != 4:
            raise CombinedAuditError("LSTM fold/seed macro metrics require four equal-weight assets.")
        row = dict(zip(groups, key, strict=True))
        row.update(
            {
                "n_assets": 4,
                "n_observations": int(frame["n"].sum()),
                **{metric: float(frame[metric].mean()) for metric in _SUMMARY_METRICS},
            }
        )
        rows.append(row)
    columns = [
        "run_id",
        "fold",
        "seed",
        "model_config_id",
        "n_assets",
        "n_observations",
        *_SUMMARY_METRICS,
    ]
    return pd.DataFrame(rows).loc[:, columns].sort_values(
        ["fold", "seed"], kind="mergesort"
    ).reset_index(drop=True)


def compute_lstm_macro_seed_summary(macro_by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (run_id, fold, model_config_id), frame in macro_by_seed.groupby(
        ["run_id", "fold", "model_config_id"], sort=True, observed=True
    ):
        if set(frame["seed"].astype(int)) != set(LSTM_SEEDS) or len(frame) != 3:
            raise CombinedAuditError(f"LSTM macro {fold} lacks exactly three fixed seeds.")
        if frame["n_observations"].nunique() != 1:
            raise CombinedAuditError(f"LSTM macro {fold} seeds have unequal row counts.")
        row: dict[str, Any] = {
            "run_id": run_id,
            "fold": fold,
            "model_config_id": model_config_id,
            "n_seeds": 3,
            "n_assets": 4,
            "n_observations_per_seed": int(frame["n_observations"].iloc[0]),
        }
        for metric in _SUMMARY_METRICS:
            values = frame[metric].to_numpy(dtype="float64")
            row.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_std": float(values.std(ddof=0)),
                    f"{metric}_min": float(values.min()),
                    f"{metric}_max": float(values.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold", kind="mergesort").reset_index(drop=True)


def recompute_all_metrics(
    baseline: pd.DataFrame, lightgbm: pd.DataFrame, lstm: pd.DataFrame
) -> RecomputedMetrics:
    baseline_asset = compute_baseline_asset_metrics(baseline)
    baseline_macro = compute_baseline_macro_metrics(baseline_asset)
    lightgbm_asset = compute_learned_asset_metrics(lightgbm, baseline, family="lightgbm")
    lightgbm_macro = compute_lightgbm_macro_metrics(lightgbm_asset)
    lstm_seed = compute_learned_asset_metrics(lstm, baseline, family="lstm")
    lstm_asset_seed_summary = compute_lstm_asset_seed_summary(lstm_seed)
    lstm_macro_by_seed = compute_lstm_macro_by_seed(lstm_seed)
    lstm_macro_seed_summary = compute_lstm_macro_seed_summary(lstm_macro_by_seed)
    return RecomputedMetrics(
        baseline_asset=baseline_asset,
        baseline_macro=baseline_macro,
        lightgbm_asset=lightgbm_asset,
        lightgbm_macro=lightgbm_macro,
        lstm_seed=lstm_seed,
        lstm_asset_seed_summary=lstm_asset_seed_summary,
        lstm_macro_by_seed=lstm_macro_by_seed,
        lstm_macro_seed_summary=lstm_macro_seed_summary,
    )


def maximum_frame_difference(
    computed: pd.DataFrame,
    saved: pd.DataFrame,
    *,
    keys: Sequence[str],
) -> float:
    if list(computed.columns) != list(saved.columns):
        raise CombinedAuditError(
            f"Saved metric schema differs. Computed={list(computed.columns)}, saved={list(saved.columns)}."
        )
    left = computed.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    right = saved.sort_values(list(keys), kind="mergesort").reset_index(drop=True)
    if len(left) != len(right):
        raise CombinedAuditError("Saved metric row count differs from independent recomputation.")
    maximum = 0.0
    for column in left.columns:
        if column in keys:
            if not left[column].astype(str).equals(right[column].astype(str)):
                raise CombinedAuditError(f"Saved metric keys differ in column {column}.")
            continue
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            left_values = left[column].to_numpy(dtype="float64")
            right_values = right[column].to_numpy(dtype="float64")
            difference = float(np.max(np.abs(left_values - right_values), initial=0.0))
            maximum = max(maximum, difference)
            if not np.allclose(
                left_values,
                right_values,
                rtol=NUMERIC_TOLERANCE,
                atol=NUMERIC_TOLERANCE,
                equal_nan=True,
            ):
                raise CombinedAuditError(f"Saved metric values differ in column {column}.")
        elif not left[column].fillna("<NA>").astype(str).equals(
            right[column].fillna("<NA>").astype(str)
        ):
            raise CombinedAuditError(f"Saved metric values differ in column {column}.")
    return maximum


def verify_saved_metrics(metrics: RecomputedMetrics) -> dict[str, float]:
    checks = {
        "baseline_asset": (metrics.baseline_asset, "baseline_asset_metrics", ["fold", "asset"]),
        "baseline_macro": (metrics.baseline_macro, "baseline_macro_metrics", ["fold"]),
        "lightgbm_asset": (
            metrics.lightgbm_asset,
            "lightgbm_asset_metrics",
            ["fold", "asset", "seed"],
        ),
        "lightgbm_macro": (metrics.lightgbm_macro, "lightgbm_macro_metrics", ["fold"]),
        "lstm_seed": (metrics.lstm_seed, "lstm_seed_metrics", ["fold", "asset", "seed"]),
        "lstm_asset_seed_summary": (
            metrics.lstm_asset_seed_summary,
            "lstm_asset_seed_summary",
            ["fold", "asset"],
        ),
        "lstm_macro_by_seed": (
            metrics.lstm_macro_by_seed,
            "lstm_macro_by_seed",
            ["fold", "seed"],
        ),
        "lstm_macro_seed_summary": (
            metrics.lstm_macro_seed_summary,
            "lstm_macro_seed_summary",
            ["fold"],
        ),
    }
    differences: dict[str, float] = {}
    for label, (computed, artifact_name, keys) in checks.items():
        saved = pd.read_csv(REPOSITORY_ROOT / FROZEN_FILES[artifact_name][0])
        differences[label] = maximum_frame_difference(computed, saved, keys=keys)
    return differences


def _add_provenance(frame: pd.DataFrame, provenance: Mapping[str, str]) -> pd.DataFrame:
    result = frame.copy()
    for column in PROVENANCE_COLUMNS:
        result[column] = provenance[column]
    return result


def build_asset_fold_comparison(metrics: RecomputedMetrics) -> pd.DataFrame:
    baseline = metrics.baseline_asset.rename(
        columns={
            "n": "n_observations",
            "mae": "baseline_mae",
            "rmse": "baseline_rmse",
            "directional_accuracy": "baseline_direction_persistence_da",
            "actual_positive_rate": "actual_positive_class_balance",
        }
    ).loc[
        :,
        [
            "fold",
            "asset",
            "n_observations",
            "baseline_mae",
            "baseline_rmse",
            "baseline_direction_persistence_da",
            "actual_positive_class_balance",
        ],
    ]
    lightgbm = metrics.lightgbm_asset.rename(
        columns={
            "model_config_id": "lightgbm_selected_config",
            "mae": "lightgbm_mae",
            "rmse": "lightgbm_rmse",
            "directional_accuracy": "lightgbm_da",
            "mae_skill": "lightgbm_mae_skill",
            "da_difference": "lightgbm_da_difference",
        }
    ).loc[
        :,
        [
            "fold",
            "asset",
            "lightgbm_selected_config",
            "lightgbm_mae",
            "lightgbm_rmse",
            "lightgbm_da",
            "lightgbm_mae_skill",
            "lightgbm_da_difference",
        ],
    ]
    lstm = metrics.lstm_asset_seed_summary.rename(
        columns={
            "model_config_id": "lstm_selected_context",
            "mae_mean": "lstm_seed_mean_mae",
            "mae_std": "lstm_seed_std_mae",
            "rmse_mean": "lstm_seed_mean_rmse",
            "directional_accuracy_mean": "lstm_seed_mean_da",
            "directional_accuracy_std": "lstm_seed_std_da",
            "mae_skill_mean": "lstm_seed_mean_mae_skill",
            "mae_skill_std": "lstm_seed_std_mae_skill",
            "da_difference_mean": "lstm_seed_mean_da_difference",
        }
    ).loc[
        :,
        [
            "fold",
            "asset",
            "lstm_selected_context",
            "lstm_seed_mean_mae",
            "lstm_seed_std_mae",
            "lstm_seed_mean_rmse",
            "lstm_seed_mean_da",
            "lstm_seed_std_da",
            "lstm_seed_mean_mae_skill",
            "lstm_seed_std_mae_skill",
            "lstm_seed_mean_da_difference",
        ],
    ]
    combined = baseline.merge(lightgbm, on=["fold", "asset"], validate="one_to_one").merge(
        lstm, on=["fold", "asset"], validate="one_to_one"
    )
    if len(combined) != 20:
        raise CombinedAuditError("Combined asset/fold table must have exactly 20 rows.")
    return combined.sort_values(["fold", "asset"], kind="mergesort").reset_index(drop=True)


def build_macro_fold_comparison(metrics: RecomputedMetrics) -> pd.DataFrame:
    baseline = metrics.baseline_macro.rename(
        columns={
            "mae": "baseline_macro_mae",
            "rmse": "baseline_macro_rmse",
            "directional_accuracy": "baseline_macro_da",
        }
    ).loc[:, ["fold", "n_observations", "baseline_macro_mae", "baseline_macro_rmse", "baseline_macro_da"]]
    lightgbm = metrics.lightgbm_macro.rename(
        columns={
            "model_config_id": "lightgbm_selected_config",
            "mae": "lightgbm_macro_mae",
            "rmse": "lightgbm_macro_rmse",
            "directional_accuracy": "lightgbm_macro_da",
            "mae_skill": "lightgbm_macro_mae_skill",
            "da_difference": "lightgbm_macro_da_difference",
        }
    ).loc[
        :,
        [
            "fold",
            "lightgbm_selected_config",
            "lightgbm_macro_mae",
            "lightgbm_macro_rmse",
            "lightgbm_macro_da",
            "lightgbm_macro_mae_skill",
            "lightgbm_macro_da_difference",
        ],
    ]
    lstm = metrics.lstm_macro_seed_summary.rename(
        columns={
            "model_config_id": "lstm_selected_context",
            "mae_mean": "lstm_seed_mean_macro_mae",
            "mae_std": "lstm_seed_std_macro_mae",
            "rmse_mean": "lstm_seed_mean_macro_rmse",
            "directional_accuracy_mean": "lstm_seed_mean_macro_da",
            "directional_accuracy_std": "lstm_seed_std_macro_da",
            "mae_skill_mean": "lstm_seed_mean_macro_mae_skill",
            "mae_skill_std": "lstm_seed_std_macro_mae_skill",
            "da_difference_mean": "lstm_seed_mean_macro_da_difference",
        }
    ).loc[
        :,
        [
            "fold",
            "lstm_selected_context",
            "lstm_seed_mean_macro_mae",
            "lstm_seed_std_macro_mae",
            "lstm_seed_mean_macro_rmse",
            "lstm_seed_mean_macro_da",
            "lstm_seed_std_macro_da",
            "lstm_seed_mean_macro_mae_skill",
            "lstm_seed_std_macro_mae_skill",
            "lstm_seed_mean_macro_da_difference",
        ],
    ]
    combined = baseline.merge(lightgbm, on="fold", validate="one_to_one").merge(
        lstm, on="fold", validate="one_to_one"
    )
    if list(combined["fold"]) != list(FOLDS):
        combined = combined.set_index("fold").loc[list(FOLDS)].reset_index()
    return combined


def build_cross_fold_summary(macro: pd.DataFrame) -> pd.DataFrame:
    series = {
        ("lightgbm", "macro_mae_skill"): macro["lightgbm_macro_mae_skill"],
        ("lightgbm", "macro_da_difference"): macro["lightgbm_macro_da_difference"],
        ("lightgbm", "macro_mae"): macro["lightgbm_macro_mae"],
        ("lightgbm", "macro_da"): macro["lightgbm_macro_da"],
        ("lstm_seed_mean", "macro_mae_skill"): macro["lstm_seed_mean_macro_mae_skill"],
        ("lstm_seed_mean", "macro_da_difference"): macro[
            "lstm_seed_mean_macro_da_difference"
        ],
        ("lstm_seed_mean", "macro_mae"): macro["lstm_seed_mean_macro_mae"],
        ("lstm_seed_mean", "macro_da"): macro["lstm_seed_mean_macro_da"],
    }
    rows = []
    for (model, metric), values in series.items():
        numeric = values.to_numpy(dtype="float64")
        q1, median, q3 = np.quantile(numeric, [0.25, 0.5, 0.75])
        rows.append(
            {
                "model": model,
                "metric": metric,
                "n_folds": len(numeric),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "minimum": float(numeric.min()),
                "maximum": float(numeric.max()),
                "interpretation": "descriptive_fold_dispersion_not_confidence_interval",
            }
        )
    return pd.DataFrame(rows)


def win_loss_counts(values: Iterable[float], *, tolerance: float = NUMERIC_TOLERANCE) -> dict[str, int]:
    numeric = np.asarray(list(values), dtype="float64")
    if numeric.ndim != 1 or not np.isfinite(numeric).all():
        raise ValueError("win/loss inputs must be a finite one-dimensional sequence")
    positive = int(np.sum(numeric > tolerance))
    negative = int(np.sum(numeric < -tolerance))
    zero = int(len(numeric) - positive - negative)
    return {"positive": positive, "zero": zero, "negative": negative, "total": len(numeric)}


def build_win_loss_table(metrics: RecomputedMetrics) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("lightgbm", "single_deterministic_seed", "all", metrics.lightgbm_asset),
        ("lstm", "three_seed_metric_mean", "mean", metrics.lstm_asset_seed_summary),
    ]
    for model, aggregation, seed, frame in comparisons:
        for metric, column in (
            ("mae_skill", "mae_skill" if model == "lightgbm" else "mae_skill_mean"),
            ("da_difference", "da_difference" if model == "lightgbm" else "da_difference_mean"),
        ):
            counts = win_loss_counts(frame[column])
            rows.append(
                {
                    "model": model,
                    "aggregation": aggregation,
                    "seed": seed,
                    "metric": metric,
                    "positive_count": counts["positive"],
                    "zero_count": counts["zero"],
                    "negative_count": counts["negative"],
                    "total_count": counts["total"],
                    "strict_numeric_tolerance": NUMERIC_TOLERANCE,
                }
            )
    for seed in LSTM_SEEDS:
        frame = metrics.lstm_seed.loc[metrics.lstm_seed["seed"] == seed]
        for metric in ("mae_skill", "da_difference"):
            counts = win_loss_counts(frame[metric])
            rows.append(
                {
                    "model": "lstm",
                    "aggregation": "individual_seed_descriptive",
                    "seed": str(seed),
                    "metric": metric,
                    "positive_count": counts["positive"],
                    "zero_count": counts["zero"],
                    "negative_count": counts["negative"],
                    "total_count": counts["total"],
                    "strict_numeric_tolerance": NUMERIC_TOLERANCE,
                }
            )
    return pd.DataFrame(rows)


def build_lstm_seed_stability(metrics: RecomputedMetrics) -> pd.DataFrame:
    summary = metrics.lstm_asset_seed_summary
    result = pd.DataFrame(
        {
            "fold": summary["fold"],
            "asset": summary["asset"],
            "lstm_selected_context": summary["model_config_id"],
            "mae_std": summary["mae_std"],
            "mae_range": summary["mae_max"] - summary["mae_min"],
            "mae_min": summary["mae_min"],
            "mae_max": summary["mae_max"],
            "mae_skill_std": summary["mae_skill_std"],
            "mae_skill_range": summary["mae_skill_max"] - summary["mae_skill_min"],
            "mae_skill_min": summary["mae_skill_min"],
            "mae_skill_max": summary["mae_skill_max"],
            "da_std": summary["directional_accuracy_std"],
            "da_range": summary["directional_accuracy_max"]
            - summary["directional_accuracy_min"],
            "da_min": summary["directional_accuracy_min"],
            "da_max": summary["directional_accuracy_max"],
            "mae_skill_sign_change": (summary["mae_skill_min"] < -NUMERIC_TOLERANCE)
            & (summary["mae_skill_max"] > NUMERIC_TOLERANCE),
        }
    )
    return result.sort_values(["fold", "asset"], kind="mergesort").reset_index(drop=True)


def build_fold_ranking(macro: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, fold_row in macro.iterrows():
        observations = [
            ("baseline", fold_row["baseline_macro_mae"], fold_row["baseline_macro_da"]),
            ("lightgbm", fold_row["lightgbm_macro_mae"], fold_row["lightgbm_macro_da"]),
            (
                "lstm_seed_mean",
                fold_row["lstm_seed_mean_macro_mae"],
                fold_row["lstm_seed_mean_macro_da"],
            ),
        ]
        mae_ranks = pd.Series({name: mae for name, mae, _ in observations}).rank(
            method="min", ascending=True
        )
        da_ranks = pd.Series({name: da for name, _, da in observations}).rank(
            method="min", ascending=False
        )
        for name, mae, da in observations:
            rows.append(
                {
                    "fold": fold_row["fold"],
                    "model": name,
                    "macro_mae": float(mae),
                    "mae_rank_primary": int(mae_ranks[name]),
                    "macro_directional_accuracy": float(da),
                    "directional_accuracy_rank_secondary": int(da_ranks[name]),
                    "ranking_policy": "MAE_primary_DA_separate_secondary_no_synthetic_score",
                }
            )
    return pd.DataFrame(rows)


def build_complexity_summary(metrics: RecomputedMetrics) -> pd.DataFrame:
    lgbm_counts = win_loss_counts(metrics.lightgbm_asset["mae_skill"])
    lstm_counts = win_loss_counts(metrics.lstm_asset_seed_summary["mae_skill_mean"])
    lgbm_macro_positive = win_loss_counts(metrics.lightgbm_macro["mae_skill"])["positive"]
    lstm_macro_positive = win_loss_counts(
        metrics.lstm_macro_seed_summary["mae_skill_mean"]
    )["positive"]
    rows = [
        {
            "model": "baseline",
            "candidate_fits": 0,
            "winner_refits": 0,
            "total_benchmark_fits": 0,
            "reported_seeds": 0,
            "deterministic_reference_execution": True,
            "initialization_variability_reported": False,
            "training_cost": "none; direct rules only",
            "observed_development_benefit": "reference comparator; skill versus self not meaningful",
        },
        {
            "model": "lightgbm",
            "candidate_fits": 80,
            "winner_refits": 20,
            "total_benchmark_fits": 100,
            "reported_seeds": 1,
            "deterministic_reference_execution": True,
            "initialization_variability_reported": False,
            "training_cost": "low relative to LSTM; 15.55 s full reference runner",
            "observed_development_benefit": (
                f"positive MAE skill in {lgbm_counts['positive']}/20 asset-fold cells and "
                f"{lgbm_macro_positive}/5 macro folds"
            ),
        },
        {
            "model": "lstm",
            "candidate_fits": 120,
            "winner_refits": 60,
            "total_benchmark_fits": 180,
            "reported_seeds": 3,
            "deterministic_reference_execution": True,
            "initialization_variability_reported": True,
            "training_cost": "substantially higher; 1064.876 s full reference runner",
            "observed_development_benefit": (
                f"positive seed-mean MAE skill in {lstm_counts['positive']}/20 asset-fold cells "
                f"and {lstm_macro_positive}/5 macro folds"
            ),
        },
    ]
    return pd.DataFrame(rows)


def validate_gate_decision(value: Mapping[str, Any]) -> None:
    gates = value.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != {f"G{index}" for index in range(1, 8)}:
        raise CombinedAuditError("PatchTST gate must include exactly G1--G7.")
    for gate_id, gate in gates.items():
        if not isinstance(gate, Mapping) or gate.get("status") not in ALLOWED_GATE_STATUSES:
            raise CombinedAuditError(f"{gate_id} has an invalid gate status.")
        if not str(gate.get("criterion", "")).strip() or not str(gate.get("evidence", "")).strip():
            raise CombinedAuditError(f"{gate_id} must include an explicit criterion and evidence.")
    decision = value.get("decision")
    if decision not in ALLOWED_GATE_DECISIONS:
        raise CombinedAuditError("PatchTST gate decision is outside the allowed enum.")
    if decision.startswith("ADMIT") and any(gate["status"] == "FAIL" for gate in gates.values()):
        raise CombinedAuditError("PatchTST cannot be admitted while a gate is FAIL.")


def build_patchtst_gate(
    metrics: RecomputedMetrics,
    *,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    lgbm_macro_counts = win_loss_counts(metrics.lightgbm_macro["mae_skill"])
    lstm_macro_counts = win_loss_counts(metrics.lstm_macro_seed_summary["mae_skill_mean"])
    gates = {
        "G1": {
            "status": "PASS",
            "criterion": "Pipeline maturity",
            "evidence": (
                "Frozen baseline, LightGBM, and LSTM artifacts passed SHA-256, full-tree, "
                "scope, exact-key, and independent metric-reproduction checks."
            ),
        },
        "G2": {
            "status": "PASS",
            "criterion": "Benchmark incompleteness",
            "evidence": (
                f"LightGBM had positive macro MAE skill in {lgbm_macro_counts['positive']}/5 "
                f"folds; LSTM seed-mean had positive macro MAE skill in "
                f"{lstm_macro_counts['positive']}/5. These inconsistent/negative results do not "
                "establish that every materially different deep time-series inductive bias is "
                "scientifically redundant; the question is diversity, not winner-seeking."
            ),
        },
        "G3": {
            "status": "PASS",
            "criterion": "Architectural relevance",
            "evidence": (
                "The frozen specification defines PatchTST as a patch-based transformer for "
                "longer temporal context, materially distinct from tabular trees and recurrent LSTM."
            ),
        },
        "G4": {
            "status": "PASS",
            "criterion": "Dataset feasibility",
            "evidence": (
                "D1 already supplied 942 causal training feature rows per asset and 880 valid "
                "63-session LSTM endpoints per asset (3,520 across four assets), with 17 frozen "
                "features. This is sufficient for a bounded pilot, though not proof that a full "
                "transformer benchmark will be worthwhile."
            ),
        },
        "G5": {
            "status": "PASS",
            "criterion": "Compute feasibility",
            "evidence": (
                "The three-seed, 180-fit CPU LSTM benchmark completed locally in 1,064.876 s. "
                "A one-fold, very-small-space CPU pilot with an explicit wall-time/memory cap is "
                "therefore practical; no PatchTST fit or dependency installation occurred here."
            ),
        },
        "G6": {
            "status": "PASS",
            "criterion": "Scope control",
            "evidence": (
                "The existing immutable target, 17 features, D1--D5 folds, artifact schema, and "
                "selection rules can be reused; Phase 2H must freeze a very small candidate space "
                "and pilot acceptance rule before any fit."
            ),
        },
        "G7": {
            "status": "PASS",
            "criterion": "Final-test protection",
            "evidence": (
                "Every current comparison is D1--D5/2020--2024 only, and the pilot can be confined "
                "to a preregistered development fold without any F1/2025 access."
            ),
        },
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "phase2g_patchtst_gate_decision",
        "gate_definition_frozen_before_decision": True,
        "gates": gates,
        "decision": "ADMIT PATCHTST TO A PREREGISTERED PILOT",
        "decision_scope": (
            "Admission authorizes only Phase 2H preregistration followed by a separately gated, "
            "bounded pilot. It does not authorize implementation in Phase 2G or a full D1--D5 run."
        ),
        "next_phase": "Phase 2H — PatchTST Preregistration and Bounded Pilot",
        "pilot_must_freeze_before_first_fit": [
            "architecture",
            "context_lengths",
            "seeds",
            "training_budget",
            "pilot_fold",
            "pilot_acceptance_criteria",
        ],
        "full_benchmark_automatic_after_pilot": False,
        "f1_2025_performance_evaluated": False,
        "regime_analysis_performed": False,
        "models_trained_during_phase2g": False,
        "patchtst_implemented": False,
        "provenance": dict(provenance),
    }
    validate_gate_decision(value)
    return value


def assert_training_free_phase2g_sources() -> dict[str, Any]:
    """Statically prove that Phase 2G source has no model/training import or call."""

    paths = [
        Path(__file__),
        Path(__file__).with_name("run_combined_audit.py"),
    ]
    forbidden_import_prefixes = (
        "src.models",
        "src.evaluation.lightgbm_evaluation",
        "src.evaluation.lstm_evaluation",
        "src.evaluation.run_lightgbm",
        "src.evaluation.run_lstm",
        "torch",
        "lightgbm",
    )
    forbidden_calls = {"fit", "fit_candidate", "refit_and_predict", "train_one_epoch", "backward"}
    inspected: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_import_prefixes):
                        raise CombinedAuditError(f"Training/model import forbidden in Phase 2G: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(forbidden_import_prefixes):
                    raise CombinedAuditError(f"Training/model import forbidden in Phase 2G: {module}")
            elif isinstance(node, ast.Call):
                call_name = ""
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                if call_name in forbidden_calls:
                    raise CombinedAuditError(f"Training call forbidden in Phase 2G: {call_name}")
        inspected.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    return {"status": "PASS", "inspected_files": inspected, "training_calls_found": 0}


def _compare_combined_frames(
    expected: pd.DataFrame, actual_path: Path, *, keys: Sequence[str]
) -> float:
    actual = pd.read_csv(actual_path)
    return maximum_frame_difference(expected, actual, keys=keys)


def run_combined_audit(*, generated_at: datetime | None = None) -> CombinedAuditResult:
    """Execute Phase 2G without importing or invoking any model training code."""

    integrity_before = verify_frozen_integrity()
    training_free = assert_training_free_phase2g_sources()

    baseline = load_prediction_artifact(REPOSITORY_ROOT / FROZEN_FILES["baseline_predictions"][0])
    lightgbm = load_prediction_artifact(REPOSITORY_ROOT / FROZEN_FILES["lightgbm_predictions"][0])
    lstm = load_prediction_artifact(REPOSITORY_ROOT / FROZEN_FILES["lstm_predictions"][0])
    validate_prediction_scope(
        baseline,
        family="baseline",
        expected_rows=5032,
        expected_regime_placeholder="not_labeled_phase2c",
    )
    validate_prediction_scope(
        lightgbm,
        family="lightgbm",
        expected_rows=5032,
        expected_regime_placeholder="not_labeled_pre_regime_analysis",
    )
    validate_prediction_scope(
        lstm,
        family="lstm",
        expected_rows=15096,
        expected_regime_placeholder="not_labeled_pre_regime_analysis",
    )
    key_audit = audit_canonical_keys(baseline, lightgbm, lstm)

    metrics = recompute_all_metrics(baseline, lightgbm, lstm)
    metric_differences = verify_saved_metrics(metrics)

    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat().replace(
        "+00:00", "Z"
    )
    source_identity = source_tree_identity()
    provenance = {
        "generation_timestamp_utc": timestamp,
        "experiment_spec_version": SPEC_VERSION,
        "experiment_spec_sha256": EXPERIMENT_SPEC_SHA256,
        "model_preregistration_path": "docs/MODEL_PREREGISTRATION.md",
        "model_preregistration_sha256": PREREGISTRATION_SHA256,
        "baseline_prediction_sha256": BASELINE_PREDICTION_SHA256,
        "lightgbm_prediction_sha256": LIGHTGBM_PREDICTION_SHA256,
        "lstm_prediction_sha256": LSTM_PREDICTION_SHA256,
        "source_tree_sha256": source_identity["sha256"],
        "source_revision": "not_available_repository_has_no_git_metadata",
    }

    plain_tables = {
        "development_asset_fold_comparison": build_asset_fold_comparison(metrics),
        "development_macro_fold_comparison": build_macro_fold_comparison(metrics),
    }
    plain_tables["cross_fold_summary"] = build_cross_fold_summary(
        plain_tables["development_macro_fold_comparison"]
    )
    plain_tables["baseline_win_loss_counts"] = build_win_loss_table(metrics)
    plain_tables["lstm_seed_stability"] = build_lstm_seed_stability(metrics)
    plain_tables["model_complexity_summary"] = build_complexity_summary(metrics)
    plain_tables["development_fold_ranking"] = build_fold_ranking(
        plain_tables["development_macro_fold_comparison"]
    )
    tables = {name: _add_provenance(frame, provenance) for name, frame in plain_tables.items()}

    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for name, table in tables.items():
        path, digest = _write_content_addressed_csv(table, name)
        artifact_paths[name] = path
        artifact_hashes[name] = digest

    gate_decision = build_patchtst_gate(metrics, provenance=provenance)
    gate_path, gate_hash = _write_content_addressed_json(gate_decision, "patchtst_gate_decision")
    artifact_paths["patchtst_gate_decision"] = gate_path
    artifact_hashes["patchtst_gate_decision"] = gate_hash

    # Independent post-write pass: reload canonical inputs and reconstruct every table again.
    baseline_second = load_prediction_artifact(
        REPOSITORY_ROOT / FROZEN_FILES["baseline_predictions"][0]
    )
    lightgbm_second = load_prediction_artifact(
        REPOSITORY_ROOT / FROZEN_FILES["lightgbm_predictions"][0]
    )
    lstm_second = load_prediction_artifact(REPOSITORY_ROOT / FROZEN_FILES["lstm_predictions"][0])
    key_audit_second = audit_canonical_keys(baseline_second, lightgbm_second, lstm_second)
    metrics_second = recompute_all_metrics(baseline_second, lightgbm_second, lstm_second)
    verify_saved_metrics(metrics_second)
    second_plain = {
        "development_asset_fold_comparison": build_asset_fold_comparison(metrics_second),
        "development_macro_fold_comparison": build_macro_fold_comparison(metrics_second),
    }
    second_plain["cross_fold_summary"] = build_cross_fold_summary(
        second_plain["development_macro_fold_comparison"]
    )
    second_plain["baseline_win_loss_counts"] = build_win_loss_table(metrics_second)
    second_plain["lstm_seed_stability"] = build_lstm_seed_stability(metrics_second)
    second_plain["model_complexity_summary"] = build_complexity_summary(metrics_second)
    second_plain["development_fold_ranking"] = build_fold_ranking(
        second_plain["development_macro_fold_comparison"]
    )
    combined_differences: dict[str, float] = {}
    combined_keys = {
        "development_asset_fold_comparison": ["fold", "asset"],
        "development_macro_fold_comparison": ["fold"],
        "cross_fold_summary": ["model", "metric"],
        "baseline_win_loss_counts": ["model", "aggregation", "seed", "metric"],
        "lstm_seed_stability": ["fold", "asset"],
        "model_complexity_summary": ["model"],
        "development_fold_ranking": ["fold", "model"],
    }
    for name, frame in second_plain.items():
        expected = _add_provenance(frame, provenance)
        combined_differences[name] = _compare_combined_frames(
            expected, artifact_paths[name], keys=combined_keys[name]
        )

    reloaded_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    validate_gate_decision(reloaded_gate)
    if reloaded_gate != gate_decision:
        raise CombinedAuditError("Reloaded PatchTST gate JSON differs from in-memory decision.")

    integrity_after = verify_frozen_integrity()
    if integrity_after["trees"] != integrity_before["trees"]:
        raise CombinedAuditError("A frozen benchmark artifact tree changed during Phase 2G.")

    comparison = plain_tables["development_asset_fold_comparison"]
    macro = plain_tables["development_macro_fold_comparison"]
    seed_stability = plain_tables["lstm_seed_stability"]
    largest_seed_row = seed_stability.loc[seed_stability["mae_std"].idxmax()]
    smallest_seed_row = seed_stability.loc[seed_stability["mae_std"].idxmin()]
    summary = {
        "asset_fold_groups": len(comparison),
        "folds": list(macro["fold"]),
        "assets": list(ASSETS),
        "lightgbm_positive_macro_mae_skill_folds": win_loss_counts(
            macro["lightgbm_macro_mae_skill"]
        )["positive"],
        "lstm_positive_macro_mae_skill_folds": win_loss_counts(
            macro["lstm_seed_mean_macro_mae_skill"]
        )["positive"],
        "largest_lstm_mae_seed_dispersion": {
            "fold": largest_seed_row["fold"],
            "asset": largest_seed_row["asset"],
            "population_std": float(largest_seed_row["mae_std"]),
            "range": float(largest_seed_row["mae_range"]),
        },
        "smallest_lstm_mae_seed_dispersion": {
            "fold": smallest_seed_row["fold"],
            "asset": smallest_seed_row["asset"],
            "population_std": float(smallest_seed_row["mae_std"]),
            "range": float(smallest_seed_row["mae_range"]),
        },
        "lstm_skill_sign_change_cells": int(seed_stability["mae_skill_sign_change"].sum()),
        "no_2025": True,
        "regime_placeholders_unchanged": True,
        "models_trained": False,
        "patchtst_implemented": False,
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "phase2g_combined_verification_manifest",
        "status": "completed_and_independently_verified",
        "generation_timestamp_utc": timestamp,
        "experiment_specification": {
            "path": "docs/EXPERIMENT_SPEC.md",
            "version": SPEC_VERSION,
            "sha256": EXPERIMENT_SPEC_SHA256,
        },
        "model_preregistration": {
            "path": "docs/MODEL_PREREGISTRATION.md",
            "sha256": PREREGISTRATION_SHA256,
        },
        "source_tree_identity": source_identity,
        "source_revision": {
            "git_revision": None,
            "status": "not_available_repository_has_no_git_metadata",
        },
        "frozen_input_integrity_before": integrity_before,
        "frozen_input_integrity_after": integrity_after,
        "canonical_key_audit": key_audit,
        "canonical_key_audit_post_write": key_audit_second,
        "metric_recomputation_max_abs_differences": metric_differences,
        "combined_post_write_max_abs_differences": combined_differences,
        "combined_artifacts": {
            name: {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": artifact_hashes[name],
                **({"rows": len(tables[name])} if name in tables else {}),
            }
            for name, path in artifact_paths.items()
        },
        "gate_decision": {
            "value": gate_decision["decision"],
            "gates": {key: value["status"] for key, value in gate_decision["gates"].items()},
        },
        "scope_guards": {
            "development_folds_only": list(FOLDS),
            "target_years_only": [2020, 2021, 2022, 2023, 2024],
            "f1_2025_performance_evaluated": False,
            "regime_analysis_performed": False,
            "regime_columns_modified": False,
            "models_trained_during_phase2g": False,
            "model_configurations_changed": False,
            "patchtst_implemented": False,
            "patchtst_dependencies_installed": False,
        },
        "training_free_static_audit": training_free,
        "summary": summary,
        "specification_deviations": [],
    }
    manifest_path, manifest_hash = _write_content_addressed_json(
        manifest, "combined_verification_manifest"
    )
    artifact_paths["combined_verification_manifest"] = manifest_path
    artifact_hashes["combined_verification_manifest"] = manifest_hash

    return CombinedAuditResult(
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        integrity=integrity_after,
        key_audit=key_audit,
        metric_differences=metric_differences,
        gate_decision=gate_decision,
        summary=summary,
    )


__all__ = [
    "ALLOWED_GATE_DECISIONS",
    "ALLOWED_GATE_STATUSES",
    "ASSETS",
    "CombinedAuditError",
    "FOLDS",
    "KEY_COLUMNS",
    "LSTM_SEEDS",
    "NUMERIC_TOLERANCE",
    "assert_training_free_phase2g_sources",
    "audit_canonical_keys",
    "build_asset_fold_comparison",
    "build_cross_fold_summary",
    "build_lstm_seed_stability",
    "build_macro_fold_comparison",
    "build_patchtst_gate",
    "build_win_loss_table",
    "compute_baseline_asset_metrics",
    "compute_baseline_macro_metrics",
    "compute_learned_asset_metrics",
    "compute_lightgbm_macro_metrics",
    "compute_lstm_asset_seed_summary",
    "compute_lstm_macro_by_seed",
    "compute_lstm_macro_seed_summary",
    "load_prediction_artifact",
    "recompute_all_metrics",
    "run_combined_audit",
    "validate_gate_decision",
    "validate_prediction_scope",
    "verify_frozen_integrity",
    "win_loss_counts",
]
