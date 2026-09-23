"""Phase 2J development-only market-regime stress-test analysis.

This module is intentionally analysis-only.  It reads frozen processed features
and frozen D1--D5 predictions, derives the preregistered SPY regime labels, and
computes descriptive evaluation tables.  It contains no model implementation,
optimizer, fitting, or prediction-generation path.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import platform
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.evaluation.walk_forward import (
    DEVELOPMENT_FOLD_IDS,
    FoldRegistry,
    load_fold_registry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPOSITORY_ROOT / "results" / "regime" / "development"
REPORT_PATH = REPOSITORY_ROOT / "PHASE_2J_REPORT.md"

SPEC_VERSION = "1.1"
FOLDS = tuple(DEVELOPMENT_FOLD_IDS)
ASSETS = ("AAPL", "MSFT", "GOOGL", "NVDA")
LSTM_SEEDS = (1729, 2718, 31415)
KEY_COLUMNS = ("fold", "asset", "origin_date", "target_date")
LABEL_JOIN_COLUMNS = ("fold", "origin_date")
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
SPY_SOURCE_COLUMNS = (
    "asset",
    "origin_date",
    "target_date",
    "spy_log_return_1d",
    "spy_volatility_21",
    "spy_trend_63",
)

DEVELOPMENT_START = date(2015, 1, 1)
DEVELOPMENT_END = date(2024, 12, 31)
NUMERIC_TOLERANCE = 1e-12
ACTUAL_RTOL = 1e-12
ACTUAL_ATOL = 1e-15
MINIMUM_HEADLINE_ROWS = 30
RV21_ANNUALIZATION = float(np.sqrt(252.0))
RV21_JUMP_THRESHOLD = 1.5
SHOCK_QUANTILE = 0.95

PROCESSED_DATA_SHA256 = (
    "9352da6b434011f9844ef0acc1813ea33dce39152712752bba21a3005aabf7be"
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
REGIME_PREREGISTRATION_SHA256 = (
    "2c0a9e9a03cb5f546bc09234ddec147ef720d03f0cddcf88fcb417d0ed341331"
)
REGIME_CONFIG_SHA256 = (
    "e6241685edf87e5e9997dd73e8bf82c34cdfc26ccb92bad2037440925100473d"
)

PROCESSED_DATA_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / f"supervised_v1_{PROCESSED_DATA_SHA256}.csv"
)
BASELINE_PREDICTION_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "baselines"
    / "predictions"
    / f"baseline_predictions_{BASELINE_PREDICTION_SHA256}.csv"
)
LIGHTGBM_PREDICTION_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "lightgbm"
    / "predictions"
    / f"lightgbm_development_predictions_{LIGHTGBM_PREDICTION_SHA256}.csv"
)
LSTM_PREDICTION_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "lstm"
    / "predictions"
    / f"lstm_development_predictions_{LSTM_PREDICTION_SHA256}.csv"
)
REGIME_PREREGISTRATION_PATH = REPOSITORY_ROOT / "docs" / "REGIME_PREREGISTRATION.md"
REGIME_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "regime_analysis.yaml"
PATCHTST_AUTHORIZATION_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "patchtst"
    / "authorization"
    / "patchtst_full_benchmark_authorization_"
    "5b63c323c9b28ddd1aa666365fb9df785fc21e6d45705317624e2ac77c811405.json"
)

FROZEN_FILES: Mapping[str, tuple[str, str]] = {
    "processed_dataset": (
        PROCESSED_DATA_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        PROCESSED_DATA_SHA256,
    ),
    "baseline_predictions": (
        BASELINE_PREDICTION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        BASELINE_PREDICTION_SHA256,
    ),
    "lightgbm_predictions": (
        LIGHTGBM_PREDICTION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        LIGHTGBM_PREDICTION_SHA256,
    ),
    "lstm_predictions": (
        LSTM_PREDICTION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        LSTM_PREDICTION_SHA256,
    ),
    "experiment_spec": (
        "docs/EXPERIMENT_SPEC.md",
        "4a29b96a7c826634d2dede63220c378737c0e545a7e733491a0a7002d69fc5b9",
    ),
    "model_preregistration": (
        "docs/MODEL_PREREGISTRATION.md",
        "be04b9ce8457c5b26a8042f548eccce8ddb4b5d1b86a1404a43ef1bb1a52d613",
    ),
    "patchtst_preregistration": (
        "docs/PATCHTST_PREREGISTRATION.md",
        "adb4be727379d005dc69fd6d8528c6e46abda993394f21a946097e59cabfb486",
    ),
    "regime_preregistration": (
        "docs/REGIME_PREREGISTRATION.md",
        REGIME_PREREGISTRATION_SHA256,
    ),
    "model_search_config": (
        "configs/model_search.yaml",
        "e15e42ff2b2f4f2bc7ccc185195c26fd35851b259bc28e08a6ba2149b6aaf38c",
    ),
    "patchtst_search_config": (
        "configs/patchtst_search.yaml",
        "6be2b9e2017c1f7b89c553ef1f65eaa995e6cd2cbc233fa2cd84a51a12c91124",
    ),
    "regime_analysis_config": (
        "configs/regime_analysis.yaml",
        REGIME_CONFIG_SHA256,
    ),
    "patchtst_authorization": (
        PATCHTST_AUTHORIZATION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "5b63c323c9b28ddd1aa666365fb9df785fc21e6d45705317624e2ac77c811405",
    ),
}

FROZEN_TREES: Mapping[str, Mapping[str, Any]] = {
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
    "combined": {
        "path": "results/combined",
        "file_count": 9,
        "sha256": "c0dc9928c7f9381bf0980d6ccd4552d1bd053d41456a19b1c992fb66a2d63be2",
    },
    "patchtst_pilot": {
        "path": "results/patchtst/pilot",
        "file_count": 34,
        "sha256": "97b630d8d9b9ab889c020152eb29dc1555fe7a33a42c99f30d1aabe0371941ce",
    },
    "patchtst_authorization": {
        "path": "results/patchtst/authorization",
        "file_count": 1,
        "sha256": "6658ee12ac703c791be435274a641457c88e2258c632c462d96b177785ab9a53",
    },
}

REGIME_VALUES: Mapping[str, tuple[str, str]] = {
    "trend": ("negative_trend", "positive_trend"),
    "volatility": ("high_volatility", "low_volatility"),
    "transition": ("transition", "stable"),
}
REGIME_COLUMN_BY_DIMENSION = {
    "trend": "trend_regime",
    "volatility": "volatility_regime",
    "transition": "transition_regime",
}
CONTRASTS: Mapping[str, tuple[str, str]] = {
    "trend": ("negative_trend", "positive_trend"),
    "volatility": ("high_volatility", "low_volatility"),
    "transition": ("transition", "stable"),
}
BASE_METRICS = (
    "mae",
    "rmse",
    "directional_accuracy",
    "actual_positive_direction_balance",
)
LEARNED_METRICS = (
    *BASE_METRICS,
    "matched_baseline_mae",
    "mae_skill",
    "matched_direction_persistence_da",
    "da_difference",
)
DEGRADATION_METRICS = (
    "mae_skill_degradation",
    "mae_raw_degradation",
    "rmse_raw_degradation",
    "da_degradation",
    "directional_stability_loss",
)


class RegimeAnalysisError(RuntimeError):
    """Raised when Phase 2J encounters a contract or integrity violation."""


@dataclass(frozen=True)
class Phase2JResult:
    artifact_paths: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]
    tables: Mapping[str, pd.DataFrame]
    manifest: Mapping[str, Any]
    integrity_before: Mapping[str, Any]
    integrity_after: Mapping[str, Any]
    spy_verification: Mapping[str, Any]
    training_free_audit: Mapping[str, Any]


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


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RegimeAnalysisError(f"Refusing to overwrite immutable artifact {path}.")
        return
    path.write_bytes(payload)


def write_content_addressed_csv(frame: pd.DataFrame, stem: str) -> tuple[Path, str]:
    payload = canonical_csv_bytes(frame)
    digest = sha256_bytes(payload)
    path = RESULT_ROOT / f"{stem}_{digest}.csv"
    _write_immutable(path, payload)
    return path, digest


def write_content_addressed_json(value: Mapping[str, Any], stem: str) -> tuple[Path, str]:
    payload = canonical_json_bytes(value)
    digest = sha256_bytes(payload)
    path = RESULT_ROOT / f"{stem}_{digest}.json"
    _write_immutable(path, payload)
    return path, digest


def tree_identity(relative_root: str) -> dict[str, Any]:
    root = (REPOSITORY_ROOT / relative_root).resolve()
    if not root.is_relative_to(REPOSITORY_ROOT.resolve()):
        raise RegimeAnalysisError(f"Tree path escaped repository: {relative_root}.")
    if not root.is_dir():
        raise RegimeAnalysisError(f"Missing frozen tree: {relative_root}.")
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
        item_hash = file_sha256(path)
        per_file[relative] = item_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "method": "sha256_of_sorted_relative_path_nul_file_sha256_lf_records",
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": per_file,
        "git_revision": None,
        "git_status": "not_available_repository_has_no_git_metadata",
    }


def ensure_no_existing_regime_artifacts() -> None:
    if RESULT_ROOT.exists() and any(path.is_file() for path in RESULT_ROOT.rglob("*")):
        raise RegimeAnalysisError(
            "A Phase 2J regime-result artifact already exists; immutable execution will not rerun."
        )


def load_regime_contract(path: Path = REGIME_CONFIG_PATH) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RegimeAnalysisError("Regime config must contain a mapping.")
    if payload.get("status") != "FROZEN BEFORE REGIME LABELING AND STRESS-TEST EXECUTION":
        raise RegimeAnalysisError("Regime config does not have the frozen Phase 2I status.")
    if tuple(payload["threshold_estimation"]["development_test"]["evaluation_folds"]) != FOLDS:
        raise RegimeAnalysisError("Regime contract development folds differ from D1--D5.")
    if tuple(payload["models"]["lstm_seeds"]) != LSTM_SEEDS:
        raise RegimeAnalysisError("Regime contract LSTM seeds changed.")
    if payload["models"]["excluded"]["patchtst"]["status"] != "CLOSED":
        raise RegimeAnalysisError("PatchTST exclusion is no longer CLOSED.")
    if payload["sample_size"]["minimum_headline_rows"] != MINIMUM_HEADLINE_ROWS:
        raise RegimeAnalysisError("Regime contract sample-size threshold changed.")
    return payload


def verify_frozen_integrity() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name, (relative, expected) in FROZEN_FILES.items():
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise RegimeAnalysisError(f"Missing frozen file: {relative}.")
        actual = file_sha256(path)
        if actual != expected:
            raise RegimeAnalysisError(
                f"Frozen file hash mismatch for {relative}: {actual} != {expected}."
            )
        files[name] = {"path": relative, "sha256": actual}

    trees: dict[str, dict[str, Any]] = {}
    for name, expected in FROZEN_TREES.items():
        actual = tree_identity(str(expected["path"]))
        if (
            actual["file_count"] != expected["file_count"]
            or actual["sha256"] != expected["sha256"]
        ):
            raise RegimeAnalysisError(
                f"Frozen {name} tree changed: {actual} != {dict(expected)}."
            )
        trees[name] = actual

    authorization = json.loads(PATCHTST_AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    if (
        authorization.get("final_authorization_decision")
        != "DECLINE_FULL_PATCHTST_BENCHMARK"
        or authorization.get("full_benchmark_authorized") is not False
    ):
        raise RegimeAnalysisError("PatchTST full benchmark is not in the frozen CLOSED state.")

    load_regime_contract()
    return {
        "status": "PASS",
        "files": files,
        "trees": trees,
        "patchtst": {
            "decision": "DECLINE_FULL_PATCHTST_BENCHMARK",
            "status": "CLOSED",
            "included_in_phase2j": False,
        },
    }


def assert_analysis_only_sources() -> dict[str, Any]:
    paths = [
        Path(__file__),
        Path(__file__).with_name("run_regime_analysis.py"),
    ]
    forbidden_import_prefixes = (
        "src.models",
        "src.evaluation.lightgbm_evaluation",
        "src.evaluation.lstm_evaluation",
        "src.evaluation.patchtst_pilot",
        "src.evaluation.run_lightgbm",
        "src.evaluation.run_lstm",
        "src.evaluation.run_patchtst_pilot",
        "lightgbm",
        "torch",
    )
    forbidden_calls = {
        "fit",
        "fit_candidate",
        "refit_and_predict",
        "train_one_epoch",
        "backward",
        "step",
    }
    inspected: list[str] = []
    for path in paths:
        if not path.is_file():
            raise RegimeAnalysisError(f"Missing Phase 2J source file: {path.name}.")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_import_prefixes):
                        raise RegimeAnalysisError(
                            f"Training/model import forbidden in Phase 2J: {alias.name}."
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(forbidden_import_prefixes):
                    raise RegimeAnalysisError(
                        f"Training/model import forbidden in Phase 2J: {module}."
                    )
            elif isinstance(node, ast.Call):
                name = ""
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                if name in forbidden_calls:
                    raise RegimeAnalysisError(
                        f"Training-like call forbidden in Phase 2J source: {name}."
                    )
        inspected.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    return {
        "status": "PASS",
        "inspected_files": inspected,
        "training_imports_found": 0,
        "training_calls_found": 0,
    }


def load_prediction_artifact(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "run_id": "string",
            "spec_version": "string",
            "data_version": "string",
            "model": "string",
            "model_config_id": "string",
            "seed": "string",
            "fold": "string",
            "partition": "string",
            "asset": "string",
            "trend_regime": "string",
            "volatility_regime": "string",
            "transition_regime": "string",
        },
    )
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise RegimeAnalysisError(
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
    expected_placeholder: str,
) -> None:
    if len(frame) != expected_rows:
        raise RegimeAnalysisError(
            f"{family} prediction rows are {len(frame)}, expected {expected_rows}."
        )
    if set(frame["fold"].astype(str)) != set(FOLDS):
        raise RegimeAnalysisError(f"{family} predictions are not exactly D1--D5.")
    if set(frame["asset"].astype(str)) != set(ASSETS):
        raise RegimeAnalysisError(f"{family} predictions do not contain the four assets.")
    if set(frame["partition"].astype(str)) != {"test"}:
        raise RegimeAnalysisError(f"{family} predictions are not test-only.")
    expected_year = {fold: 2020 + i for i, fold in enumerate(FOLDS)}
    for fold, group in frame.groupby("fold", observed=True):
        if set(group["target_date"].dt.year) != {expected_year[str(fold)]}:
            raise RegimeAnalysisError(f"{family} {fold} target-date assignment changed.")
    if (frame["target_date"].dt.date > DEVELOPMENT_END).any():
        raise RegimeAnalysisError(f"{family} contains a post-2024 row.")
    if not np.isfinite(
        frame[["actual_log_return", "predicted_log_return"]].to_numpy(dtype="float64")
    ).all():
        raise RegimeAnalysisError(f"{family} contains a non-finite metric input.")
    expected_actual_direction = (frame["actual_log_return"] > 0).astype("int8")
    if not np.array_equal(frame["actual_direction"], expected_actual_direction):
        raise RegimeAnalysisError(f"{family} actual direction is not strict-sign derived.")
    if not frame["actual_direction"].isin([0, 1]).all() or not frame[
        "predicted_direction"
    ].isin([0, 1]).all():
        raise RegimeAnalysisError(f"{family} contains a non-binary direction.")
    for column in REGIME_COLUMNS:
        if set(frame[column].astype(str)) != {expected_placeholder}:
            raise RegimeAnalysisError(f"{family} {column} placeholder changed.")

    duplicate_keys = list(KEY_COLUMNS) + (["seed"] if family == "lstm" else [])
    if frame.duplicated(duplicate_keys, keep=False).any():
        raise RegimeAnalysisError(f"{family} contains duplicate canonical keys.")
    if family == "baseline":
        if set(frame["seed"].astype(str)) != {"not_applicable"}:
            raise RegimeAnalysisError("Baseline seed identity changed.")
        if not np.array_equal(frame["predicted_log_return"], np.zeros(len(frame))):
            raise RegimeAnalysisError("Baseline zero-return predictions changed.")
    else:
        seeds = set(pd.to_numeric(frame["seed"], errors="raise").astype(int))
        expected_seeds = {1729} if family == "lightgbm" else set(LSTM_SEEDS)
        if seeds != expected_seeds:
            raise RegimeAnalysisError(f"{family} seeds changed: {seeds}.")
        expected_predicted_direction = (frame["predicted_log_return"] > 0).astype("int8")
        if not np.array_equal(frame["predicted_direction"], expected_predicted_direction):
            raise RegimeAnalysisError(
                f"{family} predicted direction is not strict-sign derived."
            )


def _sorted_keys(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[:, KEY_COLUMNS]
        .sort_values(list(KEY_COLUMNS), kind="mergesort")
        .reset_index(drop=True)
    )


def audit_prediction_keys(
    baseline: pd.DataFrame,
    lightgbm: pd.DataFrame,
    lstm: pd.DataFrame,
) -> dict[str, Any]:
    reference = _sorted_keys(baseline)
    if not reference.equals(_sorted_keys(lightgbm)):
        raise RegimeAnalysisError("LightGBM keys differ from baseline; intersection is forbidden.")
    maximum_differences: dict[str, float] = {}
    for label, frame in [
        ("lightgbm", lightgbm),
        *[
            (
                f"lstm_seed_{seed}",
                lstm.loc[pd.to_numeric(lstm["seed"]) == seed].copy(),
            )
            for seed in LSTM_SEEDS
        ],
    ]:
        if not reference.equals(_sorted_keys(frame)):
            raise RegimeAnalysisError(f"{label} keys differ from baseline.")
        merged = baseline.loc[:, [*KEY_COLUMNS, "actual_log_return", "actual_direction"]].merge(
            frame.loc[:, [*KEY_COLUMNS, "actual_log_return", "actual_direction"]],
            on=list(KEY_COLUMNS),
            how="outer",
            validate="one_to_one",
            suffixes=("_baseline", "_candidate"),
            indicator=True,
        )
        if set(merged["_merge"].astype(str)) != {"both"}:
            raise RegimeAnalysisError(f"{label} exact actual-value match is incomplete.")
        left = merged["actual_log_return_baseline"].to_numpy(dtype="float64")
        right = merged["actual_log_return_candidate"].to_numpy(dtype="float64")
        if not np.allclose(left, right, rtol=ACTUAL_RTOL, atol=ACTUAL_ATOL):
            raise RegimeAnalysisError(f"{label} actual returns differ from baseline.")
        if not np.array_equal(
            merged["actual_direction_baseline"], merged["actual_direction_candidate"]
        ):
            raise RegimeAnalysisError(f"{label} actual directions differ from baseline.")
        maximum_differences[label] = float(np.max(np.abs(left - right), initial=0.0))
    return {
        "status": "PASS",
        "base_keys": len(reference),
        "key_columns": list(KEY_COLUMNS),
        "exact_key_equality": True,
        "no_intersection_used": True,
        "maximum_actual_log_return_difference": maximum_differences,
    }


def _parse_optional_float(value: str) -> float:
    if value is None or value.strip() == "":
        return float("nan")
    result = float(value)
    return result


def load_development_spy_source(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load only permitted 2015--2024 SPY fields with an immediate date firewall.

    The CSV parser necessarily scans later bytes.  For each row, however, this
    function reads and parses target_date first.  When it is later than 2024,
    the row is discarded before any asset, origin, feature, or target value is
    accessed.  No count or diagnostic about discarded protected rows is kept.
    """

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(SPY_SOURCE_COLUMNS).issubset(
            set(reader.fieldnames)
        ):
            raise RegimeAnalysisError("Processed CSV lacks required SPY source fields.")
        for raw in reader:
            target_date = date.fromisoformat(raw["target_date"])
            if target_date > DEVELOPMENT_END:
                continue
            if target_date < DEVELOPMENT_START:
                continue
            records.append(
                {
                    "asset": raw["asset"],
                    "origin_date": pd.Timestamp(date.fromisoformat(raw["origin_date"])),
                    "target_date": pd.Timestamp(target_date),
                    "spy_log_return_1d": _parse_optional_float(
                        raw["spy_log_return_1d"]
                    ),
                    "spy_volatility_21": _parse_optional_float(
                        raw["spy_volatility_21"]
                    ),
                    "spy_trend_63": _parse_optional_float(raw["spy_trend_63"]),
                }
            )
    frame = pd.DataFrame.from_records(records, columns=SPY_SOURCE_COLUMNS)
    if frame.empty:
        raise RegimeAnalysisError("Development SPY source view is empty.")
    if set(frame["asset"]) != set(ASSETS):
        raise RegimeAnalysisError("Development source does not contain exactly four target assets.")
    if (
        frame["target_date"].dt.date.min() < DEVELOPMENT_START
        or frame["target_date"].dt.date.max() > DEVELOPMENT_END
    ):
        raise RegimeAnalysisError("A row outside 2015--2024 entered the SPY source view.")
    return frame


def _identical_repeated_values(values: pd.Series) -> bool:
    array = values.to_numpy(dtype="float64")
    if np.isnan(array).all():
        return True
    if np.isnan(array).any():
        return False
    return bool(
        np.allclose(
            array,
            np.repeat(array[0], len(array)),
            rtol=0.0,
            atol=NUMERIC_TOLERANCE,
        )
    )


def construct_unique_spy_view(
    repeated_source: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = set(SPY_SOURCE_COLUMNS)
    if not required.issubset(repeated_source.columns):
        raise RegimeAnalysisError("Repeated SPY source is missing required columns.")
    if repeated_source.duplicated(["origin_date", "asset"], keep=False).any():
        raise RegimeAnalysisError("An origin date has duplicate target-asset source rows.")

    rows: list[dict[str, Any]] = []
    for origin_date, group in repeated_source.groupby(
        "origin_date", sort=True, observed=True
    ):
        if set(group["asset"].astype(str)) != set(ASSETS) or len(group) != len(ASSETS):
            raise RegimeAnalysisError(
                f"Origin {origin_date} does not have exactly one row per target asset."
            )
        if group["target_date"].nunique(dropna=False) != 1:
            raise RegimeAnalysisError(
                f"Origin {origin_date} maps to multiple target dates/partitions."
            )
        for column in ("spy_log_return_1d", "spy_volatility_21", "spy_trend_63"):
            if not _identical_repeated_values(group[column]):
                raise RegimeAnalysisError(
                    f"Repeated SPY feature {column} disagrees across assets at {origin_date}."
                )
        first = group.iloc[0]
        rows.append(
            {
                "origin_date": pd.Timestamp(origin_date),
                "target_date": pd.Timestamp(first["target_date"]),
                "spy_log_return_1d": float(first["spy_log_return_1d"]),
                "spy_volatility_21": float(first["spy_volatility_21"]),
                "spy_trend_63": float(first["spy_trend_63"]),
            }
        )
    unique = pd.DataFrame(rows).sort_values("origin_date", kind="mergesort").reset_index(
        drop=True
    )
    if unique["origin_date"].duplicated().any() or unique["target_date"].duplicated().any():
        raise RegimeAnalysisError("Unique SPY view has duplicate origin or target dates.")
    if not unique["origin_date"].is_monotonic_increasing:
        raise RegimeAnalysisError("Unique SPY origins are not chronological.")
    if not (unique["origin_date"] < unique["target_date"]).all():
        raise RegimeAnalysisError("SPY source violates origin-before-target chronology.")

    returns = unique["spy_log_return_1d"].astype("float64")
    reconstructed_rv21 = returns.rolling(21, min_periods=21).std(ddof=0)
    reconstructed_trend63 = returns.rolling(63, min_periods=63).sum()
    verification: dict[str, Any] = {
        "status": "PASS",
        "repeated_rows": len(repeated_source),
        "unique_origin_rows": len(unique),
        "target_asset_weight_per_origin": len(ASSETS),
        "asset_feature_equality_tolerance": NUMERIC_TOLERANCE,
        "post_2024_rows_retained": 0,
    }
    for label, reconstructed, stored in (
        ("rv21", reconstructed_rv21, unique["spy_volatility_21"]),
        ("trend63", reconstructed_trend63, unique["spy_trend_63"]),
    ):
        overlap = reconstructed.notna() & stored.notna()
        if not overlap.any():
            raise RegimeAnalysisError(f"No finite overlap for reconstructed {label}.")
        left = reconstructed.loc[overlap].to_numpy(dtype="float64")
        right = stored.loc[overlap].to_numpy(dtype="float64")
        if not np.allclose(left, right, rtol=0.0, atol=NUMERIC_TOLERANCE):
            raise RegimeAnalysisError(f"Stored {label} differs from causal reconstruction.")
        if (reconstructed.notna() & stored.isna()).any():
            raise RegimeAnalysisError(f"Stored {label} becomes missing after reconstruction is valid.")
        verification[f"{label}_finite_overlap"] = int(overlap.sum())
        verification[f"{label}_max_abs_difference"] = float(
            np.max(np.abs(left - right), initial=0.0)
        )

    unique["rv21_reconstructed_raw"] = reconstructed_rv21
    unique["trend63_reconstructed"] = reconstructed_trend63
    unique["rv21_annualized"] = unique["spy_volatility_21"] * RV21_ANNUALIZATION
    unique["rv21_annualized_lag_21"] = unique["rv21_annualized"].shift(21)
    denominator = unique["rv21_annualized_lag_21"]
    numerator = unique["rv21_annualized"]
    ratio_available = (
        np.isfinite(numerator.to_numpy(dtype="float64"))
        & np.isfinite(denominator.to_numpy(dtype="float64"))
        & denominator.to_numpy(dtype="float64").__gt__(0.0)
    )
    ratio = np.full(len(unique), np.nan, dtype="float64")
    ratio[ratio_available] = (
        numerator.to_numpy(dtype="float64")[ratio_available]
        / denominator.to_numpy(dtype="float64")[ratio_available]
    )
    if np.isinf(ratio).any():
        raise RegimeAnalysisError("RV21 jump ratio created infinity.")
    unique["rv21_jump_ratio"] = ratio
    unique["rv21_jump_ratio_available"] = ratio_available
    unique["abs_spy_return"] = returns.abs()
    return unique, verification


def compute_fold_thresholds(
    unique_spy: pd.DataFrame,
    *,
    registry: FoldRegistry | None = None,
) -> pd.DataFrame:
    active_registry = registry or load_fold_registry()
    rows: list[dict[str, Any]] = []
    target_dates = pd.to_datetime(unique_spy["target_date"], errors="raise")
    for fold_id in FOLDS:
        fold = active_registry.get(fold_id)
        permitted = unique_spy.loc[
            target_dates.between(
                pd.Timestamp(fold.train.start),
                pd.Timestamp(fold.validation.end),
                inclusive="both",
            )
        ].copy()
        if permitted.empty:
            raise RegimeAnalysisError(f"{fold_id} threshold population is empty.")
        if fold.test.contains(permitted["target_date"]).any():
            raise RegimeAnalysisError(f"{fold_id} test row entered threshold history.")
        rv_values = permitted["rv21_annualized"].to_numpy(dtype="float64")
        abs_values = permitted["abs_spy_return"].to_numpy(dtype="float64")
        finite_rv = rv_values[np.isfinite(rv_values)]
        finite_abs = abs_values[np.isfinite(abs_values)]
        if len(finite_rv) == 0 or len(finite_abs) == 0:
            raise RegimeAnalysisError(f"{fold_id} has no finite threshold values.")
        rows.append(
            {
                "fold": fold_id,
                "train_target_date_start": fold.train.start.isoformat(),
                "train_target_date_end": fold.train.end.isoformat(),
                "validation_target_date_start": fold.validation.start.isoformat(),
                "validation_target_date_end": fold.validation.end.isoformat(),
                "permitted_target_date_start": fold.train.start.isoformat(),
                "permitted_target_date_end": fold.validation.end.isoformat(),
                "permitted_origin_date_start": permitted["origin_date"].min().date().isoformat(),
                "permitted_origin_date_end": permitted["origin_date"].max().date().isoformat(),
                "unique_permitted_origin_count": len(permitted),
                "finite_rv21_count": len(finite_rv),
                "volatility_median_threshold": float(np.median(finite_rv)),
                "finite_abs_return_count": len(finite_abs),
                "abs_return_q95_threshold": float(
                    pd.Series(finite_abs).quantile(0.95, interpolation="linear")
                ),
                "quantile_probability": SHOCK_QUANTILE,
                "quantile_interpolation": "linear",
                "source_population_unit": "one_unique_spy_observation_per_origin_date",
                "same_fold_test_rows_included": 0,
                "post_2024_rows_included": 0,
                "source_data_sha256": PROCESSED_DATA_SHA256,
                "regime_config_sha256": REGIME_CONFIG_SHA256,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 5 or tuple(result["fold"]) != FOLDS:
        raise RegimeAnalysisError("Threshold audit must contain exactly D1--D5.")
    return result


def prediction_label_keys(baseline: pd.DataFrame) -> pd.DataFrame:
    keys = baseline.loc[:, ["fold", "origin_date", "target_date"]].drop_duplicates()
    if keys.duplicated(["fold", "origin_date"], keep=False).any():
        raise RegimeAnalysisError("A fold/origin prediction key maps to multiple targets.")
    expected = baseline.groupby(["fold", "origin_date"], observed=True).size()
    if not expected.eq(len(ASSETS)).all():
        raise RegimeAnalysisError("A prediction fold/origin does not have all four assets.")
    return keys.sort_values(["fold", "origin_date"], kind="mergesort").reset_index(drop=True)


def classify_trend(value: float) -> str:
    if not np.isfinite(value):
        return "unavailable"
    return "positive_trend" if value > 0.0 else "negative_trend"


def classify_volatility(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        return "unavailable"
    if not np.isfinite(threshold):
        raise RegimeAnalysisError("Volatility threshold is not finite.")
    return "high_volatility" if value > threshold else "low_volatility"


def transition_components(
    *,
    rv21_current: float,
    rv21_lagged: float,
    abs_spy_return: float,
    shock_threshold: float,
) -> dict[str, Any]:
    current_available = bool(np.isfinite(rv21_current))
    denominator_available = bool(np.isfinite(rv21_lagged) and rv21_lagged > 0.0)
    ratio_available = current_available and denominator_available
    ratio = (
        float(rv21_current / rv21_lagged)
        if ratio_available
        else float("nan")
    )
    if np.isinf(ratio):
        raise RegimeAnalysisError("Transition ratio must never be infinite.")
    return_available = bool(np.isfinite(abs_spy_return))
    ratio_trigger = bool(ratio_available and ratio >= RV21_JUMP_THRESHOLD)
    return_trigger = bool(
        return_available
        and np.isfinite(shock_threshold)
        and abs_spy_return > shock_threshold
    )
    any_available = ratio_available or return_available
    if ratio_trigger or return_trigger:
        label = "transition"
    elif any_available:
        label = "stable"
    else:
        label = "unavailable"
    return {
        "rv21_current_available": current_available,
        "rv21_lag_denominator_valid": denominator_available,
        "rv21_jump_ratio_available": ratio_available,
        "rv21_jump_ratio": ratio,
        "abs_return_available": return_available,
        "transition_ratio_trigger": ratio_trigger,
        "transition_return_trigger": return_trigger,
        "transition_any_component_available": any_available,
        "transition_regime": label,
    }


def generate_regime_labels(
    label_keys: pd.DataFrame,
    unique_spy: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    source_columns = [
        "origin_date",
        "target_date",
        "spy_log_return_1d",
        "spy_volatility_21",
        "spy_trend_63",
        "rv21_annualized",
        "rv21_annualized_lag_21",
    ]
    merged = label_keys.merge(
        unique_spy.loc[:, source_columns],
        on="origin_date",
        how="left",
        validate="many_to_one",
        suffixes=("_prediction", "_source"),
        indicator=True,
    )
    if set(merged["_merge"].astype(str)) != {"both"}:
        raise RegimeAnalysisError("A development prediction origin lacks a SPY source row.")
    # CSV and Python construction can produce equivalent timestamps at different
    # NumPy resolutions (for example microseconds versus seconds).  Normalize the
    # resolution before the exact value comparison; the first dry run exposed
    # this representation-only mismatch before any result artifact was written.
    prediction_targets = pd.to_datetime(
        merged["target_date_prediction"], errors="raise"
    ).to_numpy(dtype="datetime64[ns]")
    source_targets = pd.to_datetime(
        merged["target_date_source"], errors="raise"
    ).to_numpy(dtype="datetime64[ns]")
    if not np.array_equal(prediction_targets, source_targets):
        raise RegimeAnalysisError("Prediction and SPY target dates disagree for an origin.")
    merged = merged.drop(columns=["_merge"])
    threshold_columns = [
        "fold",
        "volatility_median_threshold",
        "abs_return_q95_threshold",
    ]
    merged = merged.merge(
        thresholds.loc[:, threshold_columns],
        on="fold",
        how="left",
        validate="many_to_one",
    )
    if merged[threshold_columns[1:]].isna().any().any():
        raise RegimeAnalysisError("A development label lacks a fold threshold.")

    rows: list[dict[str, Any]] = []
    for item in merged.itertuples(index=False):
        trend_value = float(item.spy_trend_63)
        rv_raw = float(item.spy_volatility_21)
        rv_annualized = float(item.rv21_annualized)
        rv_lagged = float(item.rv21_annualized_lag_21)
        abs_return = (
            abs(float(item.spy_log_return_1d))
            if np.isfinite(float(item.spy_log_return_1d))
            else float("nan")
        )
        transition = transition_components(
            rv21_current=rv_annualized,
            rv21_lagged=rv_lagged,
            abs_spy_return=abs_return,
            shock_threshold=float(item.abs_return_q95_threshold),
        )
        rows.append(
            {
                "fold": str(item.fold),
                "origin_date": pd.Timestamp(item.origin_date),
                "target_date": pd.Timestamp(item.target_date_prediction),
                "trend63_source_value": trend_value,
                "trend_source_available": bool(np.isfinite(trend_value)),
                "trend_regime": classify_trend(trend_value),
                "rv21_raw_source_value": rv_raw,
                "rv21_annualized": rv_annualized,
                "volatility_source_available": bool(np.isfinite(rv_annualized)),
                "volatility_threshold": float(item.volatility_median_threshold),
                "volatility_regime": classify_volatility(
                    rv_annualized, float(item.volatility_median_threshold)
                ),
                "rv21_annualized_lag_21": rv_lagged,
                "rv21_current_available": transition["rv21_current_available"],
                "rv21_lag_denominator_valid": transition[
                    "rv21_lag_denominator_valid"
                ],
                "rv21_jump_ratio_available": transition["rv21_jump_ratio_available"],
                "rv21_jump_ratio": transition["rv21_jump_ratio"],
                "abs_spy_return": abs_return,
                "abs_return_available": transition["abs_return_available"],
                "shock_threshold": float(item.abs_return_q95_threshold),
                "transition_ratio_trigger": transition["transition_ratio_trigger"],
                "transition_return_trigger": transition["transition_return_trigger"],
                "transition_any_component_available": transition[
                    "transition_any_component_available"
                ],
                "transition_regime": transition["transition_regime"],
            }
        )
    labels = pd.DataFrame(rows).sort_values(
        ["fold", "origin_date"], kind="mergesort"
    ).reset_index(drop=True)
    if labels.duplicated(["fold", "origin_date"], keep=False).any():
        raise RegimeAnalysisError("Regime label artifact is not unique by fold/origin.")
    if set(labels["fold"]) != set(FOLDS):
        raise RegimeAnalysisError("Regime label artifact is not D1--D5 only.")
    if (labels["target_date"].dt.date > DEVELOPMENT_END).any():
        raise RegimeAnalysisError("A 2025 label was generated.")
    return labels


def build_label_counts(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        fold_rows = labels.loc[labels["fold"] == fold]
        for dimension, valid_values in REGIME_VALUES.items():
            column = REGIME_COLUMN_BY_DIMENSION[dimension]
            for value in (*valid_values, "unavailable"):
                count = int((fold_rows[column] == value).sum())
                rows.append(
                    {
                        "fold": fold,
                        "regime_dimension": dimension,
                        "regime_value": value,
                        "availability_status": (
                            "unavailable" if value == "unavailable" else "available"
                        ),
                        "count": count,
                    }
                )
    return pd.DataFrame(rows)


def join_regime_labels(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    mapping = labels.set_index(["fold", "origin_date"])
    if not mapping.index.is_unique:
        raise RegimeAnalysisError("Regime join mapping is not unique.")
    keys = pd.MultiIndex.from_arrays(
        [predictions["fold"], predictions["origin_date"]],
        names=["fold", "origin_date"],
    )
    missing = keys.difference(mapping.index)
    extra = mapping.index.difference(keys.unique())
    if len(missing) or len(extra):
        raise RegimeAnalysisError(
            "Regime and prediction join keys differ; intersection workaround is prohibited."
        )
    output = predictions.copy(deep=True)
    for column in REGIME_COLUMNS:
        output[column] = mapping[column].reindex(keys).to_numpy()
        if output[column].isna().any():
            raise RegimeAnalysisError(f"Regime join left a missing {column}.")
    if len(output) != len(predictions):
        raise RegimeAnalysisError("Regime join changed prediction row count.")
    if not _sorted_keys(output).equals(_sorted_keys(predictions)):
        raise RegimeAnalysisError("Regime join changed prediction keys.")
    non_regime = [column for column in PREDICTION_COLUMNS if column not in REGIME_COLUMNS]
    if not output.loc[:, non_regime].equals(predictions.loc[:, non_regime]):
        raise RegimeAnalysisError("Regime join changed a non-regime prediction field.")
    return output.loc[:, PREDICTION_COLUMNS]


def verify_cross_model_label_consistency(
    baseline: pd.DataFrame,
    lightgbm: pd.DataFrame,
    lstm: pd.DataFrame,
) -> dict[str, Any]:
    reference = (
        baseline.loc[:, ["fold", "origin_date", *REGIME_COLUMNS]]
        .drop_duplicates()
        .sort_values(["fold", "origin_date"])
        .reset_index(drop=True)
    )
    for label, frame in [
        ("lightgbm", lightgbm),
        *[
            (
                f"lstm_seed_{seed}",
                lstm.loc[pd.to_numeric(lstm["seed"]) == seed],
            )
            for seed in LSTM_SEEDS
        ],
    ]:
        candidate = (
            frame.loc[:, ["fold", "origin_date", *REGIME_COLUMNS]]
            .drop_duplicates()
            .sort_values(["fold", "origin_date"])
            .reset_index(drop=True)
        )
        if not reference.equals(candidate):
            raise RegimeAnalysisError(f"Regime labels differ for {label}.")
    grouped = lstm.groupby(list(KEY_COLUMNS), sort=False, observed=True)
    for column in REGIME_COLUMNS:
        if not grouped[column].nunique(dropna=False).eq(1).all():
            raise RegimeAnalysisError(f"LSTM seeds disagree on {column}.")
    return {
        "status": "PASS",
        "shared_fold_origin_labels": len(reference),
        "same_across_assets_models_and_lstm_seeds": True,
    }


def sample_size_status(n: int) -> str:
    return "reportable" if int(n) >= MINIMUM_HEADLINE_ROWS else "descriptive_low_n"


def compute_metric_primitives(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute the frozen row-level metric primitives for one regime cell."""

    n = len(frame)
    if n == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "directional_accuracy": float("nan"),
            "actual_positive_direction_balance": float("nan"),
            "sample_size_status": sample_size_status(0),
        }
    required = [
        "actual_log_return",
        "predicted_log_return",
        "actual_direction",
        "predicted_direction",
    ]
    if not np.isfinite(frame[required].to_numpy(dtype="float64")).all():
        raise RegimeAnalysisError("A required regime-cell metric input is non-finite.")
    actual = frame["actual_log_return"].to_numpy(dtype="float64")
    predicted = frame["predicted_log_return"].to_numpy(dtype="float64")
    actual_direction = frame["actual_direction"].to_numpy(dtype="int8")
    predicted_direction = frame["predicted_direction"].to_numpy(dtype="int8")
    error = predicted - actual
    return {
        "n": n,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "directional_accuracy": float(np.mean(predicted_direction == actual_direction)),
        "actual_positive_direction_balance": float(np.mean(actual_direction == 1)),
        "sample_size_status": sample_size_status(n),
    }


def compute_learned_metric_primitives(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute learned-model metrics after an exact canonical baseline join."""

    metrics = compute_metric_primitives(frame)
    if metrics["n"] == 0:
        metrics.update(
            {
                "matched_baseline_mae": float("nan"),
                "mae_skill": float("nan"),
                "matched_direction_persistence_da": float("nan"),
                "da_difference": float("nan"),
            }
        )
        return metrics
    required = [
        "predicted_log_return_baseline",
        "predicted_direction_baseline",
    ]
    if not np.isfinite(frame[required].to_numpy(dtype="float64")).all():
        raise RegimeAnalysisError("A matched baseline metric input is non-finite.")
    actual = frame["actual_log_return"].to_numpy(dtype="float64")
    baseline_prediction = frame["predicted_log_return_baseline"].to_numpy(
        dtype="float64"
    )
    actual_direction = frame["actual_direction"].to_numpy(dtype="int8")
    baseline_direction = frame["predicted_direction_baseline"].to_numpy(dtype="int8")
    baseline_mae = float(np.mean(np.abs(baseline_prediction - actual)))
    baseline_da = float(np.mean(baseline_direction == actual_direction))
    metrics.update(
        {
            "matched_baseline_mae": baseline_mae,
            "mae_skill": (
                float(1.0 - float(metrics["mae"]) / baseline_mae)
                if baseline_mae != 0.0
                else float("nan")
            ),
            "matched_direction_persistence_da": baseline_da,
            "da_difference": float(metrics["directional_accuracy"]) - baseline_da,
        }
    )
    return metrics


def _expected_cell_iterator() -> Iterable[tuple[str, str, str, str]]:
    for fold in FOLDS:
        for asset in ASSETS:
            for dimension, values in REGIME_VALUES.items():
                for value in values:
                    yield fold, asset, dimension, value


def compute_baseline_regime_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return all 120 explicit baseline fold/asset/regime cells."""

    rows: list[dict[str, Any]] = []
    model_values = predictions["model"].astype(str).unique()
    if len(model_values) != 1:
        raise RegimeAnalysisError("Baseline predictions contain multiple model identities.")
    config_values = predictions["model_config_id"].astype(str).unique()
    if len(config_values) != 1:
        raise RegimeAnalysisError("Baseline predictions contain multiple configurations.")
    for fold, asset, dimension, value in _expected_cell_iterator():
        regime_column = REGIME_COLUMN_BY_DIMENSION[dimension]
        cell = predictions.loc[
            (predictions["fold"] == fold)
            & (predictions["asset"] == asset)
            & (predictions[regime_column] == value)
        ]
        rows.append(
            {
                "model": "persistence_baseline",
                "source_model": model_values[0],
                "model_config_id": config_values[0],
                "seed": "not_applicable",
                "fold": fold,
                "asset": asset,
                "regime_dimension": dimension,
                "regime_value": value,
                **compute_metric_primitives(cell),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 120:
        raise RegimeAnalysisError("Baseline regime metrics must contain 120 cells.")
    return result


def match_exact_baseline_rows(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    family: str,
) -> pd.DataFrame:
    """Match every learned row to the saved Phase 2C baseline without intersection."""

    if family not in {"lightgbm", "lstm"}:
        raise ValueError("family must be lightgbm or lstm")
    baseline_columns = [
        *KEY_COLUMNS,
        "actual_log_return",
        "actual_direction",
        "predicted_log_return",
        "predicted_direction",
        *REGIME_COLUMNS,
    ]
    if baseline.duplicated(list(KEY_COLUMNS), keep=False).any():
        raise RegimeAnalysisError("Baseline contains duplicate exact-match keys.")
    if family == "lightgbm":
        if not _sorted_keys(predictions).equals(_sorted_keys(baseline)):
            raise RegimeAnalysisError("LightGBM baseline match keys are not exactly equal.")
    else:
        for seed in LSTM_SEEDS:
            seed_rows = predictions.loc[pd.to_numeric(predictions["seed"]) == seed]
            if not _sorted_keys(seed_rows).equals(_sorted_keys(baseline)):
                raise RegimeAnalysisError(
                    f"LSTM seed {seed} baseline match keys are not exactly equal."
                )
    merged = predictions.merge(
        baseline.loc[:, baseline_columns],
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one" if family == "lightgbm" else "many_to_one",
        suffixes=("", "_baseline"),
        indicator=True,
    )
    if set(merged["_merge"].astype(str)) != {"both"}:
        raise RegimeAnalysisError(f"{family} has an unmatched exact baseline row.")
    left = merged["actual_log_return"].to_numpy(dtype="float64")
    right = merged["actual_log_return_baseline"].to_numpy(dtype="float64")
    if not np.allclose(left, right, rtol=ACTUAL_RTOL, atol=ACTUAL_ATOL):
        raise RegimeAnalysisError(f"{family} actual returns differ from exact baseline rows.")
    if not np.array_equal(
        merged["actual_direction"].to_numpy(dtype="int8"),
        merged["actual_direction_baseline"].to_numpy(dtype="int8"),
    ):
        raise RegimeAnalysisError(
            f"{family} actual directions differ from exact baseline rows."
        )
    for column in REGIME_COLUMNS:
        if not merged[column].astype(str).equals(
            merged[f"{column}_baseline"].astype(str)
        ):
            raise RegimeAnalysisError(f"{family} {column} differs from matched baseline.")
    return merged.drop(columns=["_merge"])


def _single_configuration(frame: pd.DataFrame, context: str) -> str:
    values = frame["model_config_id"].astype(str).unique()
    if len(values) != 1:
        raise RegimeAnalysisError(f"{context} does not have one model configuration.")
    return str(values[0])


def compute_learned_regime_metrics(
    predictions: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    family: str,
) -> pd.DataFrame:
    """Return 120 LightGBM cells or 360 independently calculated LSTM cells."""

    matched = match_exact_baseline_rows(predictions, baseline, family=family)
    seeds = (1729,) if family == "lightgbm" else LSTM_SEEDS
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        for asset in ASSETS:
            for seed in seeds:
                realization = matched.loc[
                    (matched["fold"] == fold)
                    & (matched["asset"] == asset)
                    & (pd.to_numeric(matched["seed"]) == seed)
                ]
                configuration = _single_configuration(
                    realization, f"{family} {fold}/{asset}/{seed}"
                )
                for dimension, values in REGIME_VALUES.items():
                    regime_column = REGIME_COLUMN_BY_DIMENSION[dimension]
                    for value in values:
                        cell = realization.loc[realization[regime_column] == value]
                        rows.append(
                            {
                                "model": family,
                                "model_config_id": configuration,
                                "seed": seed,
                                "fold": fold,
                                "asset": asset,
                                "regime_dimension": dimension,
                                "regime_value": value,
                                **compute_learned_metric_primitives(cell),
                            }
                        )
    result = pd.DataFrame(rows)
    expected_rows = 120 if family == "lightgbm" else 360
    if len(result) != expected_rows:
        raise RegimeAnalysisError(
            f"{family} regime metrics contain {len(result)} rows, expected {expected_rows}."
        )
    return result


def _metric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype="float64")
    if np.isnan(array).all():
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    if not np.isfinite(array).all():
        raise RegimeAnalysisError("Seed metric summary mixes finite and undefined values.")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def compute_lstm_seed_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize three independently calculated LSTM metric values per cell."""

    rows: list[dict[str, Any]] = []
    group_columns = [
        "fold",
        "asset",
        "regime_dimension",
        "regime_value",
    ]
    for key, group in seed_metrics.groupby(group_columns, sort=True, observed=True):
        if set(group["seed"].astype(int)) != set(LSTM_SEEDS) or len(group) != 3:
            raise RegimeAnalysisError(f"LSTM cell {key} lacks exactly three fixed seeds.")
        if group["n"].nunique() != 1:
            raise RegimeAnalysisError(f"LSTM cell {key} has unequal seed row counts.")
        configurations = group["model_config_id"].astype(str).unique()
        if len(configurations) != 1:
            raise RegimeAnalysisError(f"LSTM cell {key} has unequal seed configurations.")
        all_reportable = bool(group["sample_size_status"].eq("reportable").all())
        row: dict[str, Any] = {
            "model": "lstm_seed_mean",
            "model_config_id": configurations[0],
            "fold": key[0],
            "asset": key[1],
            "regime_dimension": key[2],
            "regime_value": key[3],
            "n_seeds": 3,
            "n_per_seed": int(group["n"].iloc[0]),
            "all_seed_cells_reportable": all_reportable,
            "sample_size_status": (
                "reportable" if all_reportable else "descriptive_low_n"
            ),
        }
        for metric in LEARNED_METRICS:
            summary = _metric_summary(group[metric].to_numpy(dtype="float64"))
            for statistic, value in summary.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        ["fold", "asset", "regime_dimension", "regime_value"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(result) != 120:
        raise RegimeAnalysisError("LSTM seed-summary regime metrics must contain 120 cells.")
    return result


def _sign_class(value: float) -> str:
    if not np.isfinite(value):
        return "unavailable"
    if value < -NUMERIC_TOLERANCE:
        return "negative"
    if value > NUMERIC_TOLERANCE:
        return "positive"
    return "zero"


def _contrast_values(stressed: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    return {
        "mae_skill_degradation": float(stressed["mae_skill"])
        - float(reference["mae_skill"]),
        "mae_raw_degradation": float(stressed["mae"]) - float(reference["mae"]),
        "rmse_raw_degradation": float(stressed["rmse"]) - float(reference["rmse"]),
        "da_degradation": float(stressed["directional_accuracy"])
        - float(reference["directional_accuracy"]),
        "directional_stability_loss": float(reference["directional_accuracy"])
        - float(stressed["directional_accuracy"]),
    }


def compute_asset_fold_degradation(
    lightgbm_metrics: pd.DataFrame,
    lstm_seed_summary: pd.DataFrame,
    lstm_seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build fixed-orientation primary degradation contrasts."""

    rows: list[dict[str, Any]] = []
    for model in ("lightgbm", "lstm_seed_mean"):
        source = lightgbm_metrics if model == "lightgbm" else lstm_seed_summary
        for fold in FOLDS:
            for asset in ASSETS:
                for dimension, (stressed_value, reference_value) in CONTRASTS.items():
                    subset = source.loc[
                        (source["fold"] == fold)
                        & (source["asset"] == asset)
                        & (source["regime_dimension"] == dimension)
                    ]
                    stressed_rows = subset.loc[subset["regime_value"] == stressed_value]
                    reference_rows = subset.loc[subset["regime_value"] == reference_value]
                    if len(stressed_rows) != 1 or len(reference_rows) != 1:
                        raise RegimeAnalysisError(
                            f"Missing primary contrast cells for {model}/{fold}/{asset}/{dimension}."
                        )
                    stressed_raw = stressed_rows.iloc[0]
                    reference_raw = reference_rows.iloc[0]
                    if model == "lightgbm":
                        stressed = {metric: stressed_raw[metric] for metric in LEARNED_METRICS}
                        reference = {
                            metric: reference_raw[metric] for metric in LEARNED_METRICS
                        }
                        stressed_n = int(stressed_raw["n"])
                        reference_n = int(reference_raw["n"])
                        stressed_status = str(stressed_raw["sample_size_status"])
                        reference_status = str(reference_raw["sample_size_status"])
                        configuration = str(stressed_raw["model_config_id"])
                    else:
                        stressed = {
                            metric: stressed_raw[f"{metric}_mean"]
                            for metric in LEARNED_METRICS
                        }
                        reference = {
                            metric: reference_raw[f"{metric}_mean"]
                            for metric in LEARNED_METRICS
                        }
                        stressed_n = int(stressed_raw["n_per_seed"])
                        reference_n = int(reference_raw["n_per_seed"])
                        stressed_status = str(stressed_raw["sample_size_status"])
                        reference_status = str(reference_raw["sample_size_status"])
                        configuration = str(stressed_raw["model_config_id"])
                    contrast_status = (
                        "reportable"
                        if stressed_status == reference_status == "reportable"
                        else "descriptive_low_n"
                    )
                    values = _contrast_values(stressed, reference)
                    seed_values = ""
                    seed_signs = ""
                    seed_sign_disagreement: bool | str = ""
                    reportable_seed_count: int | str = ""
                    if model == "lstm_seed_mean":
                        diagnostics: list[float] = []
                        signs: list[str] = []
                        for seed in LSTM_SEEDS:
                            seed_subset = lstm_seed_metrics.loc[
                                (lstm_seed_metrics["fold"] == fold)
                                & (lstm_seed_metrics["asset"] == asset)
                                & (lstm_seed_metrics["seed"].astype(int) == seed)
                                & (
                                    lstm_seed_metrics["regime_dimension"]
                                    == dimension
                                )
                            ]
                            seed_stressed = seed_subset.loc[
                                seed_subset["regime_value"] == stressed_value
                            ].iloc[0]
                            seed_reference = seed_subset.loc[
                                seed_subset["regime_value"] == reference_value
                            ].iloc[0]
                            if (
                                seed_stressed["sample_size_status"] == "reportable"
                                and seed_reference["sample_size_status"] == "reportable"
                            ):
                                value = float(seed_stressed["mae_skill"]) - float(
                                    seed_reference["mae_skill"]
                                )
                                diagnostics.append(value)
                                signs.append(_sign_class(value))
                        reportable_seed_count = len(diagnostics)
                        seed_values = "|".join(f"{value:.17g}" for value in diagnostics)
                        seed_signs = "|".join(signs)
                        seed_sign_disagreement = bool(
                            len(signs) == len(LSTM_SEEDS) and len(set(signs)) > 1
                        )
                    rows.append(
                        {
                            "model": model,
                            "model_config_id": configuration,
                            "fold": fold,
                            "asset": asset,
                            "regime_dimension": dimension,
                            "stressed_regime_value": stressed_value,
                            "reference_regime_value": reference_value,
                            "stressed_n": stressed_n,
                            "reference_n": reference_n,
                            "stressed_status": stressed_status,
                            "reference_status": reference_status,
                            "contrast_status": contrast_status,
                            "headline_reportable": contrast_status == "reportable",
                            "stressed_mae": stressed["mae"],
                            "reference_mae": reference["mae"],
                            "stressed_rmse": stressed["rmse"],
                            "reference_rmse": reference["rmse"],
                            "stressed_directional_accuracy": stressed[
                                "directional_accuracy"
                            ],
                            "reference_directional_accuracy": reference[
                                "directional_accuracy"
                            ],
                            "stressed_mae_skill": stressed["mae_skill"],
                            "reference_mae_skill": reference["mae_skill"],
                            **values,
                            "lstm_reportable_seed_count": reportable_seed_count,
                            "lstm_seed_mae_skill_degradations": seed_values,
                            "lstm_seed_signs": seed_signs,
                            "lstm_seed_sign_disagreement": seed_sign_disagreement,
                        }
                    )
    result = pd.DataFrame(rows)
    if len(result) != 120:
        raise RegimeAnalysisError("Asset/fold degradation table must contain 120 rows.")
    if not np.allclose(
        result["directional_stability_loss"],
        -result["da_degradation"],
        rtol=0.0,
        atol=NUMERIC_TOLERANCE,
        equal_nan=True,
    ):
        raise RegimeAnalysisError("Directional degradation orientation changed.")
    return result


def _macro_from_asset_rows(
    rows: pd.DataFrame,
    *,
    metrics: Sequence[str],
) -> dict[str, Any]:
    if set(rows["asset"].astype(str)) != set(ASSETS) or len(rows) != len(ASSETS):
        raise RegimeAnalysisError("Macro input does not contain exactly four assets.")
    reportable = rows["sample_size_status"].eq("reportable")
    result: dict[str, Any] = {
        "n_assets_required": 4,
        "n_assets_reportable": int(reportable.sum()),
        "n_per_asset_min": int(rows["n"].min()),
        "n_per_asset_max": int(rows["n"].max()),
        "macro_status": (
            "reportable" if bool(reportable.all()) else "unavailable_incomplete_assets"
        ),
        "macro_weighting": "equal_asset_weight",
        "mae_skill_aggregation": "arithmetic_mean_of_asset_level_skills",
    }
    for metric in metrics:
        result[metric] = (
            float(rows[metric].mean()) if bool(reportable.all()) else float("nan")
        )
    return result


def compute_macro_regime_metrics(
    baseline_metrics: pd.DataFrame,
    lightgbm_metrics: pd.DataFrame,
    lstm_seed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compute complete-four-asset macro values and LSTM seed summaries."""

    rows: list[dict[str, Any]] = []
    for model, source, seeds, metrics in (
        ("persistence_baseline", baseline_metrics, ("not_applicable",), BASE_METRICS),
        ("lightgbm", lightgbm_metrics, (1729,), LEARNED_METRICS),
        ("lstm", lstm_seed_metrics, LSTM_SEEDS, LEARNED_METRICS),
    ):
        for seed in seeds:
            seed_source = (
                source
                if model == "persistence_baseline"
                else source.loc[pd.to_numeric(source["seed"]) == int(seed)]
            )
            for fold in FOLDS:
                for dimension, values in REGIME_VALUES.items():
                    for value in values:
                        asset_rows = seed_source.loc[
                            (seed_source["fold"] == fold)
                            & (seed_source["regime_dimension"] == dimension)
                            & (seed_source["regime_value"] == value)
                        ]
                        macro = _macro_from_asset_rows(asset_rows, metrics=metrics)
                        rows.append(
                            {
                                "model": model,
                                "aggregation": (
                                    "per_seed" if model == "lstm" else "deterministic"
                                ),
                                "seed": seed,
                                "fold": fold,
                                "regime_dimension": dimension,
                                "regime_value": value,
                                **macro,
                                **{
                                    metric: (
                                        macro[metric]
                                        if metric in metrics
                                        else float("nan")
                                    )
                                    for metric in LEARNED_METRICS
                                },
                            }
                        )

    direct = pd.DataFrame(rows)
    lstm_direct = direct.loc[
        (direct["model"] == "lstm") & (direct["aggregation"] == "per_seed")
    ]
    summaries: list[dict[str, Any]] = []
    for key, group in lstm_direct.groupby(
        ["fold", "regime_dimension", "regime_value"],
        sort=True,
        observed=True,
    ):
        if set(pd.to_numeric(group["seed"]).astype(int)) != set(LSTM_SEEDS):
            raise RegimeAnalysisError(f"LSTM macro cell {key} lacks three seeds.")
        all_reportable = bool(group["macro_status"].eq("reportable").all())
        row: dict[str, Any] = {
            "model": "lstm_seed_mean",
            "aggregation": "seed_metric_summary",
            "seed": "seed_mean",
            "fold": key[0],
            "regime_dimension": key[1],
            "regime_value": key[2],
            "n_assets_required": 4,
            "n_assets_reportable": (
                4 if all_reportable else int(group["n_assets_reportable"].min())
            ),
            "n_per_asset_min": int(group["n_per_asset_min"].min()),
            "n_per_asset_max": int(group["n_per_asset_max"].max()),
            "macro_status": (
                "reportable" if all_reportable else "unavailable_incomplete_assets"
            ),
            "macro_weighting": "equal_asset_weight_then_seed_metric_mean",
            "mae_skill_aggregation": "asset_skill_mean_per_seed_then_seed_mean",
        }
        for metric in LEARNED_METRICS:
            if all_reportable:
                summary = _metric_summary(group[metric].to_numpy(dtype="float64"))
            else:
                summary = _metric_summary([float("nan")] * 3)
            row[metric] = summary["mean"]
            row[f"{metric}_seed_std"] = summary["std"]
            row[f"{metric}_seed_min"] = summary["min"]
            row[f"{metric}_seed_max"] = summary["max"]
        summaries.append(row)
    result = pd.concat([direct, pd.DataFrame(summaries)], ignore_index=True, sort=False)
    expected_rows = 30 + 30 + 90 + 30
    if len(result) != expected_rows:
        raise RegimeAnalysisError(
            f"Macro regime table has {len(result)} rows, expected {expected_rows}."
        )
    return result


def compute_macro_degradation(macro_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute primary LightGBM and LSTM seed-mean macro contrasts."""

    rows: list[dict[str, Any]] = []
    selections = {
        "lightgbm": ("lightgbm", "deterministic"),
        "lstm_seed_mean": ("lstm_seed_mean", "seed_metric_summary"),
    }
    for output_model, (source_model, aggregation) in selections.items():
        source = macro_metrics.loc[
            (macro_metrics["model"] == source_model)
            & (macro_metrics["aggregation"] == aggregation)
        ]
        for fold in FOLDS:
            for dimension, (stressed_value, reference_value) in CONTRASTS.items():
                subset = source.loc[
                    (source["fold"] == fold)
                    & (source["regime_dimension"] == dimension)
                ]
                stressed_rows = subset.loc[subset["regime_value"] == stressed_value]
                reference_rows = subset.loc[subset["regime_value"] == reference_value]
                if len(stressed_rows) != 1 or len(reference_rows) != 1:
                    raise RegimeAnalysisError(
                        f"Missing macro contrast cells for {output_model}/{fold}/{dimension}."
                    )
                stressed = stressed_rows.iloc[0]
                reference = reference_rows.iloc[0]
                status = (
                    "reportable"
                    if stressed["macro_status"] == reference["macro_status"] == "reportable"
                    else "unavailable_incomplete_assets"
                )
                values = _contrast_values(stressed, reference)
                seed_signs = ""
                seed_disagreement: bool | str = ""
                if output_model == "lstm_seed_mean":
                    signs: list[str] = []
                    for seed in LSTM_SEEDS:
                        seed_subset = macro_metrics.loc[
                            (macro_metrics["model"] == "lstm")
                            & (macro_metrics["aggregation"] == "per_seed")
                            & (pd.to_numeric(macro_metrics["seed"], errors="coerce") == seed)
                            & (macro_metrics["fold"] == fold)
                            & (macro_metrics["regime_dimension"] == dimension)
                        ]
                        seed_stressed = seed_subset.loc[
                            seed_subset["regime_value"] == stressed_value
                        ].iloc[0]
                        seed_reference = seed_subset.loc[
                            seed_subset["regime_value"] == reference_value
                        ].iloc[0]
                        if (
                            seed_stressed["macro_status"] == "reportable"
                            and seed_reference["macro_status"] == "reportable"
                        ):
                            signs.append(
                                _sign_class(
                                    float(seed_stressed["mae_skill"])
                                    - float(seed_reference["mae_skill"])
                                )
                            )
                    seed_signs = "|".join(signs)
                    seed_disagreement = bool(
                        len(signs) == len(LSTM_SEEDS) and len(set(signs)) > 1
                    )
                rows.append(
                    {
                        "model": output_model,
                        "fold": fold,
                        "regime_dimension": dimension,
                        "stressed_regime_value": stressed_value,
                        "reference_regime_value": reference_value,
                        "stressed_macro_status": stressed["macro_status"],
                        "reference_macro_status": reference["macro_status"],
                        "contrast_status": status,
                        "headline_reportable": status == "reportable",
                        **values,
                        "lstm_seed_signs": seed_signs,
                        "lstm_seed_sign_disagreement": seed_disagreement,
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 30:
        raise RegimeAnalysisError("Macro degradation table must contain 30 rows.")
    return result


def compute_cross_fold_summary(macro_degradation: pd.DataFrame) -> pd.DataFrame:
    """Summarize one reportable macro value per fold with linear quartiles."""

    rows: list[dict[str, Any]] = []
    for model in ("lightgbm", "lstm_seed_mean"):
        for dimension in REGIME_VALUES:
            subset = macro_degradation.loc[
                (macro_degradation["model"] == model)
                & (macro_degradation["regime_dimension"] == dimension)
                & (macro_degradation["contrast_status"] == "reportable")
            ]
            for metric in DEGRADATION_METRICS:
                values = subset[metric].dropna().astype("float64")
                count = len(values)
                rows.append(
                    {
                        "model": model,
                        "regime_dimension": dimension,
                        "degradation_metric": metric,
                        "reportable_fold_count": count,
                        "median": (
                            float(values.median()) if count else float("nan")
                        ),
                        "q1": (
                            float(values.quantile(0.25, interpolation="linear"))
                            if count
                            else float("nan")
                        ),
                        "q3": (
                            float(values.quantile(0.75, interpolation="linear"))
                            if count
                            else float("nan")
                        ),
                        "iqr": (
                            float(
                                values.quantile(0.75, interpolation="linear")
                                - values.quantile(0.25, interpolation="linear")
                            )
                            if count
                            else float("nan")
                        ),
                        "minimum": (
                            float(values.min()) if count else float("nan")
                        ),
                        "maximum": (
                            float(values.max()) if count else float("nan")
                        ),
                        "quantile_interpolation": "linear",
                        "complete_five_fold_summary": count == 5,
                        "summary_status": (
                            "complete_headline"
                            if count == 5
                            else "descriptive_incomplete"
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 30:
        raise RegimeAnalysisError("Cross-fold summary must contain 30 rows.")
    return result


def compute_failure_patterns(
    asset_degradation: pd.DataFrame,
    macro_degradation: pd.DataFrame,
) -> pd.DataFrame:
    """Count preregistered MAE-skill degradation signs without a composite score."""

    rows: list[dict[str, Any]] = []
    for level, source in (
        ("asset_fold", asset_degradation),
        ("macro_fold", macro_degradation),
    ):
        for model in ("lightgbm", "lstm_seed_mean"):
            for dimension in REGIME_VALUES:
                subset = source.loc[
                    (source["model"] == model)
                    & (source["regime_dimension"] == dimension)
                ]
                status_column = (
                    "contrast_status" if level == "asset_fold" else "contrast_status"
                )
                reportable = subset.loc[subset[status_column] == "reportable"]
                signs = reportable["mae_skill_degradation"].map(_sign_class)
                row = {
                    "reporting_level": level,
                    "model": model,
                    "regime_dimension": dimension,
                    "total_contrasts": len(subset),
                    "reportable_contrasts": len(reportable),
                    "descriptive_or_unavailable_contrasts": len(subset)
                    - len(reportable),
                    "negative_degradation_count": int((signs == "negative").sum()),
                    "zero_within_tolerance_count": int((signs == "zero").sum()),
                    "positive_degradation_count": int((signs == "positive").sum()),
                    "sign_tolerance": NUMERIC_TOLERANCE,
                    "lstm_seed_sign_disagreement_count": (
                        int(
                            reportable["lstm_seed_sign_disagreement"]
                            .astype("boolean")
                            .fillna(False)
                            .sum()
                        )
                        if model == "lstm_seed_mean"
                        else 0
                    ),
                }
                if (
                    row["negative_degradation_count"]
                    + row["zero_within_tolerance_count"]
                    + row["positive_degradation_count"]
                    != row["reportable_contrasts"]
                ):
                    raise RegimeAnalysisError("Failure-pattern sign counts do not reconcile.")
                rows.append(row)
    return pd.DataFrame(rows)


def compute_directional_error_disagreement(
    asset_degradation: pd.DataFrame,
) -> pd.DataFrame:
    """Classify reportable asset/fold error-versus-direction degradation disagreement."""

    rows: list[dict[str, Any]] = []
    reportable = asset_degradation.loc[
        asset_degradation["contrast_status"] == "reportable"
    ]
    for item in reportable.itertuples(index=False):
        skill_worse = bool(item.mae_skill_degradation < -NUMERIC_TOLERANCE)
        direction_worse = bool(item.da_degradation < -NUMERIC_TOLERANCE)
        if skill_worse and not direction_worse:
            disagreement_type = "mae_skill_worse_direction_not_worse"
        elif direction_worse and not skill_worse:
            disagreement_type = "direction_worse_mae_skill_not_worse"
        else:
            disagreement_type = "no_disagreement"
        rows.append(
            {
                "model": item.model,
                "fold": item.fold,
                "asset": item.asset,
                "regime_dimension": item.regime_dimension,
                "mae_skill_degradation": item.mae_skill_degradation,
                "da_degradation": item.da_degradation,
                "mae_skill_worse_in_stressed_regime": skill_worse,
                "directional_accuracy_worse_in_stressed_regime": direction_worse,
                "is_disagreement": skill_worse != direction_worse,
                "disagreement_type": disagreement_type,
                "sign_tolerance": NUMERIC_TOLERANCE,
            }
        )
    return pd.DataFrame(rows)


def _load_validate_and_label() -> dict[str, Any]:
    """Complete every pre-metric gate without writing an artifact."""

    integrity = verify_frozen_integrity()
    training_free = assert_analysis_only_sources()
    baseline = load_prediction_artifact(BASELINE_PREDICTION_PATH)
    lightgbm = load_prediction_artifact(LIGHTGBM_PREDICTION_PATH)
    lstm = load_prediction_artifact(LSTM_PREDICTION_PATH)
    validate_prediction_scope(
        baseline,
        family="baseline",
        expected_rows=5032,
        expected_placeholder="not_labeled_phase2c",
    )
    validate_prediction_scope(
        lightgbm,
        family="lightgbm",
        expected_rows=5032,
        expected_placeholder="not_labeled_pre_regime_analysis",
    )
    validate_prediction_scope(
        lstm,
        family="lstm",
        expected_rows=15096,
        expected_placeholder="not_labeled_pre_regime_analysis",
    )
    key_audit = audit_prediction_keys(baseline, lightgbm, lstm)

    repeated_spy = load_development_spy_source()
    unique_spy, spy_verification = construct_unique_spy_view(repeated_spy)
    thresholds = compute_fold_thresholds(unique_spy)
    labels = generate_regime_labels(
        prediction_label_keys(baseline),
        unique_spy,
        thresholds,
    )
    label_counts = build_label_counts(labels)
    labeled_baseline = join_regime_labels(baseline, labels)
    labeled_lightgbm = join_regime_labels(lightgbm, labels)
    labeled_lstm = join_regime_labels(lstm, labels)
    label_consistency = verify_cross_model_label_consistency(
        labeled_baseline,
        labeled_lightgbm,
        labeled_lstm,
    )
    return {
        "integrity": integrity,
        "training_free": training_free,
        "key_audit": key_audit,
        "spy_verification": spy_verification,
        "unique_spy": unique_spy,
        "threshold_audit": thresholds,
        "unique_regime_labels": labels,
        "regime_label_counts": label_counts,
        "labeled_baseline_predictions": labeled_baseline,
        "labeled_lightgbm_predictions": labeled_lightgbm,
        "labeled_lstm_predictions": labeled_lstm,
        "label_consistency": label_consistency,
    }


def _calculate_metric_tables(prepared: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    baseline_metrics = compute_baseline_regime_metrics(
        prepared["labeled_baseline_predictions"]
    )
    lightgbm_metrics = compute_learned_regime_metrics(
        prepared["labeled_lightgbm_predictions"],
        prepared["labeled_baseline_predictions"],
        family="lightgbm",
    )
    lstm_seed_metrics = compute_learned_regime_metrics(
        prepared["labeled_lstm_predictions"],
        prepared["labeled_baseline_predictions"],
        family="lstm",
    )
    lstm_summary = compute_lstm_seed_summary(lstm_seed_metrics)
    degradation = compute_asset_fold_degradation(
        lightgbm_metrics,
        lstm_summary,
        lstm_seed_metrics,
    )
    macro = compute_macro_regime_metrics(
        baseline_metrics,
        lightgbm_metrics,
        lstm_seed_metrics,
    )
    macro_degradation = compute_macro_degradation(macro)
    cross_fold = compute_cross_fold_summary(macro_degradation)
    failure_patterns = compute_failure_patterns(degradation, macro_degradation)
    disagreement = compute_directional_error_disagreement(degradation)
    return {
        "baseline_regime_metrics": baseline_metrics,
        "lightgbm_regime_metrics": lightgbm_metrics,
        "lstm_per_seed_regime_metrics": lstm_seed_metrics,
        "lstm_seed_summary_regime_metrics": lstm_summary,
        "asset_fold_degradation": degradation,
        "macro_regime_metrics": macro,
        "macro_degradation": macro_degradation,
        "cross_fold_summary": cross_fold,
        "failure_pattern_summary": failure_patterns,
        "directional_error_disagreement": disagreement,
    }


TABLE_STEMS: Mapping[str, str] = {
    "threshold_audit": "regime_threshold_audit",
    "unique_regime_labels": "development_unique_regime_labels",
    "regime_label_counts": "development_regime_label_counts",
    "labeled_baseline_predictions": "development_labeled_baseline_predictions",
    "labeled_lightgbm_predictions": "development_labeled_lightgbm_predictions",
    "labeled_lstm_predictions": "development_labeled_lstm_predictions",
    "baseline_regime_metrics": "baseline_regime_cell_metrics",
    "lightgbm_regime_metrics": "lightgbm_regime_cell_metrics",
    "lstm_per_seed_regime_metrics": "lstm_per_seed_regime_cell_metrics",
    "lstm_seed_summary_regime_metrics": "lstm_seed_summary_regime_metrics",
    "asset_fold_degradation": "asset_fold_regime_degradation",
    "macro_regime_metrics": "macro_regime_metrics",
    "macro_degradation": "macro_regime_degradation",
    "cross_fold_summary": "cross_fold_regime_summary",
    "failure_pattern_summary": "regime_failure_pattern_summary",
    "directional_error_disagreement": "directional_error_disagreement",
}


def _verify_content_addressed_path(path: Path) -> str:
    match = re.search(r"_([0-9a-f]{64})\.[^.]+$", path.name)
    if match is None:
        raise RegimeAnalysisError(f"Artifact is not content-addressed: {path.name}.")
    expected = match.group(1)
    actual = file_sha256(path)
    if actual != expected:
        raise RegimeAnalysisError(
            f"Artifact filename hash differs from bytes for {path.name}."
        )
    return actual


def _canonical_roundtrip(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(canonical_csv_bytes(frame)))


def _compare_saved_frame(expected: pd.DataFrame, path: Path) -> float:
    actual = pd.read_csv(path)
    normalized = _canonical_roundtrip(expected)
    if tuple(actual.columns) != tuple(normalized.columns) or len(actual) != len(normalized):
        raise RegimeAnalysisError(f"Saved table structure differs for {path.name}.")
    maximum = 0.0
    for column in actual.columns:
        left = normalized[column]
        right = actual[column]
        left_numeric = pd.to_numeric(left, errors="coerce")
        right_numeric = pd.to_numeric(right, errors="coerce")
        left_nonmissing = left.notna()
        right_nonmissing = right.notna()
        numeric_column = bool(
            left_numeric[left_nonmissing].notna().all()
            and right_numeric[right_nonmissing].notna().all()
            and (left_nonmissing == right_nonmissing).all()
        )
        if numeric_column:
            left_values = left_numeric.to_numpy(dtype="float64")
            right_values = right_numeric.to_numpy(dtype="float64")
            if not np.allclose(
                left_values,
                right_values,
                rtol=0.0,
                atol=NUMERIC_TOLERANCE,
                equal_nan=True,
            ):
                raise RegimeAnalysisError(
                    f"Saved numerical column {column} differs in {path.name}."
                )
            finite = np.isfinite(left_values) & np.isfinite(right_values)
            if finite.any():
                maximum = max(
                    maximum,
                    float(np.max(np.abs(left_values[finite] - right_values[finite]))),
                )
        else:
            left_text = left.astype("string").fillna("<NA>")
            right_text = right.astype("string").fillna("<NA>")
            if not left_text.equals(right_text):
                raise RegimeAnalysisError(
                    f"Saved categorical column {column} differs in {path.name}."
                )
    return maximum


def _verify_saved_label_formulas(labels: pd.DataFrame) -> None:
    if labels.duplicated(["fold", "origin_date"], keep=False).any():
        raise RegimeAnalysisError("Saved label table has duplicate fold/origin keys.")
    if set(labels["fold"].astype(str)) != set(FOLDS):
        raise RegimeAnalysisError("Saved labels are not exactly D1--D5.")
    target_dates = pd.to_datetime(labels["target_date"], errors="raise")
    if (target_dates.dt.date > DEVELOPMENT_END).any():
        raise RegimeAnalysisError("Saved labels include a post-2024 target.")
    boolean_columns = [
        "transition_ratio_trigger",
        "transition_return_trigger",
        "transition_any_component_available",
    ]
    for column in boolean_columns:
        if labels[column].dtype != bool:
            labels[column] = labels[column].astype("boolean").astype(bool)
    for item in labels.itertuples(index=False):
        if classify_trend(float(item.trend63_source_value)) != item.trend_regime:
            raise RegimeAnalysisError("Saved trend formula does not reproduce.")
        if (
            classify_volatility(
                float(item.rv21_annualized),
                float(item.volatility_threshold),
            )
            != item.volatility_regime
        ):
            raise RegimeAnalysisError("Saved volatility formula does not reproduce.")
        transition = transition_components(
            rv21_current=float(item.rv21_annualized),
            rv21_lagged=float(item.rv21_annualized_lag_21),
            abs_spy_return=float(item.abs_spy_return),
            shock_threshold=float(item.shock_threshold),
        )
        for key in (
            "transition_ratio_trigger",
            "transition_return_trigger",
            "transition_regime",
        ):
            if transition[key] != getattr(item, key):
                raise RegimeAnalysisError(f"Saved transition formula differs for {key}.")


def _verify_non_regime_prediction_values(
    saved: pd.DataFrame,
    original: pd.DataFrame,
    *,
    family: str,
) -> float:
    """Compare serialized prediction values with type-appropriate strict rules."""

    non_regime = [column for column in PREDICTION_COLUMNS if column not in REGIME_COLUMNS]
    if tuple(saved.loc[:, non_regime].columns) != tuple(
        original.loc[:, non_regime].columns
    ) or len(saved) != len(original):
        raise RegimeAnalysisError(f"Saved {family} non-regime structure changed.")
    maximum = 0.0
    for column in non_regime:
        if column in {"origin_date", "target_date"}:
            left = pd.to_datetime(saved[column], errors="raise").to_numpy(
                dtype="datetime64[ns]"
            )
            right = pd.to_datetime(original[column], errors="raise").to_numpy(
                dtype="datetime64[ns]"
            )
            if not np.array_equal(left, right):
                raise RegimeAnalysisError(f"Saved {family} {column} changed.")
        elif column in {"actual_log_return", "predicted_log_return"}:
            left = saved[column].to_numpy(dtype="float64")
            right = original[column].to_numpy(dtype="float64")
            if not np.allclose(left, right, rtol=ACTUAL_RTOL, atol=ACTUAL_ATOL):
                raise RegimeAnalysisError(f"Saved {family} {column} changed.")
            maximum = max(
                maximum,
                float(np.max(np.abs(left - right), initial=0.0)),
            )
        elif column in {"actual_direction", "predicted_direction"}:
            if not np.array_equal(
                saved[column].to_numpy(dtype="int8"),
                original[column].to_numpy(dtype="int8"),
            ):
                raise RegimeAnalysisError(f"Saved {family} {column} changed.")
        else:
            if not saved[column].astype("string").equals(
                original[column].astype("string")
            ):
                raise RegimeAnalysisError(f"Saved {family} {column} changed.")
    return maximum


def independently_verify_saved_artifacts(
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Freshly reload sources and independently reproduce all sixteen CSV tables."""

    for path in artifact_paths.values():
        _verify_content_addressed_path(path)

    fresh = _load_validate_and_label()
    recomputed: dict[str, pd.DataFrame] = {
        key: fresh[key] for key in TABLE_STEMS if key in fresh
    }
    recomputed.update(_calculate_metric_tables(fresh))
    differences: dict[str, float] = {}
    for name in TABLE_STEMS:
        if name not in artifact_paths or name not in recomputed:
            raise RegimeAnalysisError(f"Independent verification lacks table {name}.")
        differences[name] = _compare_saved_frame(
            recomputed[name],
            artifact_paths[name],
        )

    saved_thresholds = pd.read_csv(artifact_paths["threshold_audit"])
    if len(saved_thresholds) != 5 or set(saved_thresholds["fold"]) != set(FOLDS):
        raise RegimeAnalysisError("Saved threshold audit is not exactly five folds.")
    if not saved_thresholds["same_fold_test_rows_included"].eq(0).all():
        raise RegimeAnalysisError("A saved threshold includes same-fold test rows.")
    if not saved_thresholds["post_2024_rows_included"].eq(0).all():
        raise RegimeAnalysisError("A saved threshold includes post-2024 rows.")
    saved_labels = pd.read_csv(artifact_paths["unique_regime_labels"])
    _verify_saved_label_formulas(saved_labels)

    saved_predictions = {
        "baseline": load_prediction_artifact(
            artifact_paths["labeled_baseline_predictions"]
        ),
        "lightgbm": load_prediction_artifact(
            artifact_paths["labeled_lightgbm_predictions"]
        ),
        "lstm": load_prediction_artifact(artifact_paths["labeled_lstm_predictions"]),
    }
    expected_rows = {"baseline": 5032, "lightgbm": 5032, "lstm": 15096}
    originals = {
        "baseline": load_prediction_artifact(BASELINE_PREDICTION_PATH),
        "lightgbm": load_prediction_artifact(LIGHTGBM_PREDICTION_PATH),
        "lstm": load_prediction_artifact(LSTM_PREDICTION_PATH),
    }
    non_regime_maximum_differences: dict[str, float] = {}
    for family, frame in saved_predictions.items():
        if len(frame) != expected_rows[family]:
            raise RegimeAnalysisError(f"Saved {family} labeled row count changed.")
        if not _sorted_keys(frame).equals(_sorted_keys(originals[family])):
            raise RegimeAnalysisError(f"Saved {family} labeled keys changed.")
        non_regime_maximum_differences[family] = _verify_non_regime_prediction_values(
            frame,
            originals[family],
            family=family,
        )
    if set(pd.to_numeric(saved_predictions["lstm"]["seed"]).astype(int)) != set(
        LSTM_SEEDS
    ):
        raise RegimeAnalysisError("Saved labeled LSTM seeds changed.")

    return {
        "schema_version": 1,
        "artifact_type": "phase2j_independent_saved_file_verification",
        "status": "PASS",
        "verification_method": (
            "fresh_disk_reload_fresh_development_source_projection_and_full_recalculation"
        ),
        "verified_primary_artifact_count": len(TABLE_STEMS),
        "content_addressed_filename_hashes": "PASS",
        "thresholds": {
            "fold_rows": 5,
            "same_fold_test_rows_included": 0,
            "post_2024_rows_included": 0,
            "median_recomputed": True,
            "linear_q95_recomputed": True,
        },
        "labels": {
            "rows": len(saved_labels),
            "unique_fold_origin": True,
            "d1_d5_only": True,
            "post_2024_rows": 0,
            "tie_and_trigger_formulas_recomputed": True,
        },
        "labeled_prediction_rows": expected_rows,
        "non_regime_prediction_fields_unchanged": True,
        "non_regime_prediction_maximum_absolute_differences": (
            non_regime_maximum_differences
        ),
        "lstm_seeds_unchanged": True,
        "metric_layers_recomputed": [
            "cell_metrics",
            "exact_matched_baseline_metrics",
            "mae_skill",
            "da_difference",
            "lstm_seed_summaries",
            "asset_fold_degradation",
            "macro_metrics",
            "macro_degradation",
            "cross_fold_summary",
            "failure_patterns",
            "directional_error_disagreement",
        ],
        "maximum_absolute_differences": differences,
        "strict_tolerance": NUMERIC_TOLERANCE,
        "models_trained": False,
        "f1_2025_performance_evaluated": False,
        "new_f1_2025_scientific_access": False,
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        records.append(
            {
                key: (
                    None
                    if (
                        isinstance(value, (float, np.floating))
                        and not np.isfinite(value)
                    )
                    else _json_default(value)
                    if isinstance(
                        value,
                        (
                            np.integer,
                            np.floating,
                            np.bool_,
                            pd.Timestamp,
                            datetime,
                            date,
                        ),
                    )
                    else value
                )
                for key, value in raw.items()
            }
        )
    return records


def _artifact_inventory(
    paths: Mapping[str, Path],
    hashes: Mapping[str, str],
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for name, path in paths.items():
        record: dict[str, Any] = {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": hashes[name],
            "filename_matches_content_hash": True,
        }
        if name in tables:
            record["rows"] = len(tables[name])
        inventory[name] = record
    return inventory


def run_regime_analysis(
    *,
    dry_run: bool = False,
    generated_at: datetime | None = None,
) -> Phase2JResult | Mapping[str, Any]:
    """Execute the frozen Phase 2J development analysis, or its write-free dry run."""

    ensure_no_existing_regime_artifacts()
    prepared = _load_validate_and_label()
    if dry_run:
        return {
            "status": "PASS",
            "mode": "dry_run",
            "artifacts_written": 0,
            "frozen_integrity": "PASS",
            "prediction_rows": {
                "baseline": len(prepared["labeled_baseline_predictions"]),
                "lightgbm": len(prepared["labeled_lightgbm_predictions"]),
                "lstm": len(prepared["labeled_lstm_predictions"]),
            },
            "threshold_folds": len(prepared["threshold_audit"]),
            "unique_spy_deduplication": "PASS",
            "labels_structurally_constructible": True,
            "d1_d5_only": True,
            "post_2024_analysis_rows": 0,
            "patchtst_included": False,
            "training_functions_imported_or_called": False,
            "performance_results_printed": False,
        }

    metric_tables = _calculate_metric_tables(prepared)
    tables: dict[str, pd.DataFrame] = {
        "threshold_audit": prepared["threshold_audit"],
        "unique_regime_labels": prepared["unique_regime_labels"],
        "regime_label_counts": prepared["regime_label_counts"],
        "labeled_baseline_predictions": prepared["labeled_baseline_predictions"],
        "labeled_lightgbm_predictions": prepared["labeled_lightgbm_predictions"],
        "labeled_lstm_predictions": prepared["labeled_lstm_predictions"],
        **metric_tables,
    }
    if set(tables) != set(TABLE_STEMS):
        raise RegimeAnalysisError("The sixteen primary artifact tables are incomplete.")

    artifact_paths: dict[str, Path] = {}
    artifact_hashes: dict[str, str] = {}
    for name, stem in TABLE_STEMS.items():
        path, digest = write_content_addressed_csv(tables[name], stem)
        artifact_paths[name] = path
        artifact_hashes[name] = digest

    verification = independently_verify_saved_artifacts(artifact_paths)
    verification_path, verification_hash = write_content_addressed_json(
        verification,
        "phase2j_independent_verification",
    )
    artifact_paths["independent_verification"] = verification_path
    artifact_hashes["independent_verification"] = verification_hash

    integrity_after = verify_frozen_integrity()
    if integrity_after["trees"] != prepared["integrity"]["trees"]:
        raise RegimeAnalysisError("A frozen historical result tree changed during Phase 2J.")
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat().replace(
        "+00:00", "Z"
    )
    labels = tables["unique_regime_labels"]
    label_counts = tables["regime_label_counts"]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "phase2j_development_regime_run_manifest",
        "status": "completed_and_independently_verified",
        "execution_timestamp_utc": timestamp,
        "experiment_specification": {
            "version": SPEC_VERSION,
            "path": "docs/EXPERIMENT_SPEC.md",
            "sha256": FROZEN_FILES["experiment_spec"][1],
        },
        "regime_preregistration": {
            "path": "docs/REGIME_PREREGISTRATION.md",
            "sha256": REGIME_PREREGISTRATION_SHA256,
        },
        "regime_config": {
            "path": "configs/regime_analysis.yaml",
            "sha256": REGIME_CONFIG_SHA256,
        },
        "processed_dataset": {
            "path": PROCESSED_DATA_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": PROCESSED_DATA_SHA256,
        },
        "canonical_predictions": {
            "baseline": {
                "path": BASELINE_PREDICTION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": BASELINE_PREDICTION_SHA256,
                "rows": 5032,
            },
            "lightgbm": {
                "path": LIGHTGBM_PREDICTION_PATH.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "sha256": LIGHTGBM_PREDICTION_SHA256,
                "rows": 5032,
            },
            "lstm": {
                "path": LSTM_PREDICTION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": LSTM_PREDICTION_SHA256,
                "rows": 15096,
            },
        },
        "patchtst": {
            "status": "CLOSED",
            "decision": "DECLINE_FULL_PATCHTST_BENCHMARK",
            "included": False,
        },
        "exact_regime_formulas": {
            "trend": "positive iff Trend63_t > 0; negative iff <= 0",
            "volatility": (
                "population RV21 ddof=0 times sqrt(252); high iff above same-fold "
                "train+validation median; equality low"
            ),
            "transition": (
                "RV21_t/RV21_t_minus_21 >= 1.5 OR abs(SPY log return) above "
                "same-fold train+validation linear q95"
            ),
            "mae_skill": "1 - model_cell_mae / exact_matched_baseline_cell_mae",
            "degradation": "stressed_value_metric - reference_value_metric",
        },
        "thresholds_by_fold": _json_records(tables["threshold_audit"]),
        "label_counts": _json_records(label_counts),
        "unavailable_label_counts": _json_records(
            label_counts.loc[label_counts["regime_value"] == "unavailable"]
        ),
        "sample_size_rule": {
            "minimum_headline_rows": MINIMUM_HEADLINE_ROWS,
            "reportable": "n >= 30",
            "descriptive_low_n": "n < 30",
            "zero_count_cells_materialized": True,
            "low_n_numerical_metrics_retained": True,
        },
        "lstm_seeds": list(LSTM_SEEDS),
        "model_row_counts": {
            "baseline": len(tables["labeled_baseline_predictions"]),
            "lightgbm": len(tables["labeled_lightgbm_predictions"]),
            "lstm": len(tables["labeled_lstm_predictions"]),
        },
        "regime_label_rows": len(labels),
        "artifacts": _artifact_inventory(
            artifact_paths,
            artifact_hashes,
            tables,
        ),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "source_tree_identity": source_tree_identity(),
        "frozen_integrity_before": prepared["integrity"],
        "frozen_integrity_after": integrity_after,
        "prediction_key_audit": prepared["key_audit"],
        "spy_source_verification": prepared["spy_verification"],
        "label_consistency": prepared["label_consistency"],
        "training_free_static_audit": prepared["training_free"],
        "scope_guards": {
            "development_folds": list(FOLDS),
            "target_years": [2020, 2021, 2022, 2023, 2024],
            "post_2024_rows_in_analysis_populations": 0,
            "f1_2025_performance_evaluated": False,
            "new_f1_2025_scientific_access": False,
            "models_trained": False,
            "historical_predictions_modified": False,
            "patchtst_included": False,
        },
        "implementation_choices_frozen_before_execution": {
            "sign_tolerance": NUMERIC_TOLERANCE,
            "explicit_zero_count_cells": True,
            "directional_error_disagreement": (
                "xor_of_mae_skill_degradation_below_negative_tolerance_and_"
                "da_degradation_below_negative_tolerance"
            ),
            "lstm_seed_disagreement": (
                "three_reportable_seed_metric_degradation_signs_not_identical"
            ),
            "macro_requires_all_four_assets": True,
            "cross_fold_complete_requires_five_folds": True,
        },
        "independent_saved_file_verification": {
            "path": verification_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": verification_hash,
            "status": verification["status"],
        },
        "warnings": [],
        "exceptions": [],
        "specification_deviations": [],
        "test_suite": {
            "status": "required_post_artifact_generation_and_recorded_in_phase_report"
        },
    }
    manifest_path, manifest_hash = write_content_addressed_json(
        manifest,
        "phase2j_run_manifest",
    )
    artifact_paths["run_manifest"] = manifest_path
    artifact_hashes["run_manifest"] = manifest_hash
    _verify_content_addressed_path(manifest_path)

    return Phase2JResult(
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        tables=tables,
        manifest=manifest,
        integrity_before=prepared["integrity"],
        integrity_after=integrity_after,
        spy_verification=prepared["spy_verification"],
        training_free_audit=prepared["training_free"],
    )


def _discover_interrupted_primary_artifacts() -> tuple[dict[str, Path], dict[str, str]]:
    """Resolve the sixteen already-written primary files after verifier interruption."""

    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, stem in TABLE_STEMS.items():
        matches = sorted(RESULT_ROOT.glob(f"{stem}_[0-9a-f]*.csv"))
        matches = [
            path
            for path in matches
            if re.search(r"_[0-9a-f]{64}\.csv$", path.name)
        ]
        if len(matches) != 1:
            raise RegimeAnalysisError(
                f"Interrupted Phase 2J recovery expected one {stem} artifact, found "
                f"{len(matches)}."
            )
        paths[name] = matches[0]
        hashes[name] = _verify_content_addressed_path(matches[0])
    other_files = [
        path
        for path in RESULT_ROOT.glob("*")
        if path.is_file() and path not in set(paths.values())
    ]
    if other_files:
        raise RegimeAnalysisError(
            "Interrupted Phase 2J recovery found unexpected non-primary result files."
        )
    return paths, hashes


def complete_interrupted_phase2j_verification(
    *,
    generated_at: datetime | None = None,
) -> Phase2JResult:
    """Resume only the stopped post-write verification/manifest/report sequence.

    This function never recalculates or rewrites a primary scientific artifact.
    It exists solely for the observed one-run verifier interruption.
    """

    artifact_paths, artifact_hashes = _discover_interrupted_primary_artifacts()
    integrity_before = verify_frozen_integrity()
    prepared = _load_validate_and_label()
    verification = independently_verify_saved_artifacts(artifact_paths)
    verification_path, verification_hash = write_content_addressed_json(
        verification,
        "phase2j_independent_verification",
    )
    artifact_paths["independent_verification"] = verification_path
    artifact_hashes["independent_verification"] = verification_hash
    integrity_after = verify_frozen_integrity()
    if integrity_after["trees"] != integrity_before["trees"]:
        raise RegimeAnalysisError("A frozen historical tree changed during verification resume.")

    tables = {
        name: pd.read_csv(path)
        for name, path in artifact_paths.items()
        if name in TABLE_STEMS
    }
    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat().replace(
        "+00:00", "Z"
    )
    label_counts = tables["regime_label_counts"]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "phase2j_development_regime_run_manifest",
        "status": "completed_and_independently_verified_after_post_write_verifier_resume",
        "execution_timestamp_utc": timestamp,
        "experiment_specification": {
            "version": SPEC_VERSION,
            "path": "docs/EXPERIMENT_SPEC.md",
            "sha256": FROZEN_FILES["experiment_spec"][1],
        },
        "regime_preregistration": {
            "path": "docs/REGIME_PREREGISTRATION.md",
            "sha256": REGIME_PREREGISTRATION_SHA256,
        },
        "regime_config": {
            "path": "configs/regime_analysis.yaml",
            "sha256": REGIME_CONFIG_SHA256,
        },
        "processed_dataset": {
            "path": PROCESSED_DATA_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": PROCESSED_DATA_SHA256,
        },
        "canonical_predictions": {
            "baseline": {
                "path": BASELINE_PREDICTION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": BASELINE_PREDICTION_SHA256,
                "rows": 5032,
            },
            "lightgbm": {
                "path": LIGHTGBM_PREDICTION_PATH.relative_to(
                    REPOSITORY_ROOT
                ).as_posix(),
                "sha256": LIGHTGBM_PREDICTION_SHA256,
                "rows": 5032,
            },
            "lstm": {
                "path": LSTM_PREDICTION_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": LSTM_PREDICTION_SHA256,
                "rows": 15096,
            },
        },
        "patchtst": {
            "status": "CLOSED",
            "decision": "DECLINE_FULL_PATCHTST_BENCHMARK",
            "included": False,
        },
        "exact_regime_formulas": {
            "trend": "positive iff Trend63_t > 0; negative iff <= 0",
            "volatility": (
                "population RV21 ddof=0 times sqrt(252); high iff above same-fold "
                "train+validation median; equality low"
            ),
            "transition": (
                "RV21_t/RV21_t_minus_21 >= 1.5 OR abs(SPY log return) above "
                "same-fold train+validation linear q95"
            ),
            "mae_skill": "1 - model_cell_mae / exact_matched_baseline_cell_mae",
            "degradation": "stressed_value_metric - reference_value_metric",
        },
        "thresholds_by_fold": _json_records(tables["threshold_audit"]),
        "label_counts": _json_records(label_counts),
        "unavailable_label_counts": _json_records(
            label_counts.loc[label_counts["regime_value"] == "unavailable"]
        ),
        "sample_size_rule": {
            "minimum_headline_rows": MINIMUM_HEADLINE_ROWS,
            "reportable": "n >= 30",
            "descriptive_low_n": "n < 30",
            "zero_count_cells_materialized": True,
            "low_n_numerical_metrics_retained": True,
        },
        "lstm_seeds": list(LSTM_SEEDS),
        "model_row_counts": {
            "baseline": len(tables["labeled_baseline_predictions"]),
            "lightgbm": len(tables["labeled_lightgbm_predictions"]),
            "lstm": len(tables["labeled_lstm_predictions"]),
        },
        "regime_label_rows": len(tables["unique_regime_labels"]),
        "artifacts": _artifact_inventory(
            artifact_paths,
            artifact_hashes,
            tables,
        ),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "source_tree_identity": source_tree_identity(),
        "frozen_integrity_before": integrity_before,
        "frozen_integrity_after": integrity_after,
        "prediction_key_audit": prepared["key_audit"],
        "spy_source_verification": prepared["spy_verification"],
        "label_consistency": prepared["label_consistency"],
        "training_free_static_audit": prepared["training_free"],
        "scope_guards": {
            "development_folds": list(FOLDS),
            "target_years": [2020, 2021, 2022, 2023, 2024],
            "post_2024_rows_in_analysis_populations": 0,
            "f1_2025_performance_evaluated": False,
            "new_f1_2025_scientific_access": False,
            "models_trained": False,
            "historical_predictions_modified": False,
            "patchtst_included": False,
        },
        "implementation_choices_frozen_before_execution": {
            "sign_tolerance": NUMERIC_TOLERANCE,
            "explicit_zero_count_cells": True,
            "directional_error_disagreement": (
                "xor_of_mae_skill_degradation_below_negative_tolerance_and_"
                "da_degradation_below_negative_tolerance"
            ),
            "lstm_seed_disagreement": (
                "three_reportable_seed_metric_degradation_signs_not_identical"
            ),
            "macro_requires_all_four_assets": True,
            "cross_fold_complete_requires_five_folds": True,
        },
        "execution_recovery": {
            "normal_analysis_command_invocations": 1,
            "primary_artifacts_written_before_stop": 16,
            "primary_artifacts_rewritten_during_resume": False,
            "stop_stage": "independent_saved_prediction_non_regime_equality_check",
            "cause": (
                "strict_float_DataFrame_equals_rejected_sub_1e-16_CSV_roundtrip_"
                "differences_despite_frozen_numeric_tolerance"
            ),
            "maximum_observed_baseline_roundtrip_difference": 9.996344030316351e-17,
            "resolution": (
                "type_appropriate_exact_categorical_and_date_checks_plus_frozen_"
                "numeric_tolerance"
            ),
            "scientific_method_changed": False,
        },
        "independent_saved_file_verification": {
            "path": verification_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": verification_hash,
            "status": verification["status"],
        },
        "warnings": [
            "Normal execution stopped after the sixteen primary immutable artifacts "
            "because its verifier used dtype-sensitive exact float equality."
        ],
        "exceptions": [
            {
                "stage": "independent_saved_file_verification",
                "resolved": True,
                "scientific_effect": "none",
                "primary_artifacts_rewritten": False,
            }
        ],
        "specification_deviations": [],
        "test_suite": {
            "status": "required_post_artifact_generation_and_recorded_in_phase_report"
        },
    }
    manifest_path, manifest_hash = write_content_addressed_json(
        manifest,
        "phase2j_run_manifest",
    )
    artifact_paths["run_manifest"] = manifest_path
    artifact_hashes["run_manifest"] = manifest_hash
    _verify_content_addressed_path(manifest_path)
    result = Phase2JResult(
        artifact_paths=artifact_paths,
        artifact_hashes=artifact_hashes,
        tables=tables,
        manifest=manifest,
        integrity_before=integrity_before,
        integrity_after=integrity_after,
        spy_verification=prepared["spy_verification"],
        training_free_audit=prepared["training_free"],
    )
    write_phase2j_report(result)
    return result


def _format_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_format_value(row[column]) for column in columns)
        + " |"
        for _, row in frame.loc[:, columns].iterrows()
    ]
    return "\n".join([header, separator, *body])


def _degradation_summary_text(
    degradation: pd.DataFrame,
    dimension: str,
) -> str:
    lines: list[str] = []
    for model in ("lightgbm", "lstm_seed_mean"):
        subset = degradation.loc[
            (degradation["model"] == model)
            & (degradation["regime_dimension"] == dimension)
            & (degradation["contrast_status"] == "reportable")
        ]
        signs = subset["mae_skill_degradation"].map(_sign_class)
        median = (
            float(subset["mae_skill_degradation"].median())
            if len(subset)
            else float("nan")
        )
        lines.append(
            f"- {model}: {len(subset)} reportable asset/fold contrasts; "
            f"{int((signs == 'negative').sum())} negative, "
            f"{int((signs == 'zero').sum())} zero within tolerance, and "
            f"{int((signs == 'positive').sum())} positive MAE-skill degradations; "
            f"median {_format_value(median)}."
        )
    return "\n".join(lines)


def build_phase2j_report(
    result: Phase2JResult,
    *,
    full_test_result: str = "FINAL_TEST_RESULT_PENDING",
) -> str:
    """Build the required 36-section Phase 2J report."""

    tables = result.tables
    thresholds = tables["threshold_audit"]
    label_counts = tables["regime_label_counts"]
    degradation = tables["asset_fold_degradation"]
    macro_degradation = tables["macro_degradation"]
    failure = tables["failure_pattern_summary"]
    disagreement = tables["directional_error_disagreement"]
    cross_fold = tables["cross_fold_summary"]

    def dimension_counts(dimension: str) -> str:
        subset = label_counts.loc[label_counts["regime_dimension"] == dimension]
        return _markdown_table(subset, ["fold", "regime_value", "count"])

    baseline_metrics = tables["baseline_regime_metrics"]
    lightgbm_metrics = tables["lightgbm_regime_metrics"]
    lstm_seed_metrics = tables["lstm_per_seed_regime_metrics"]
    lstm_summary = tables["lstm_seed_summary_regime_metrics"]
    macro = tables["macro_regime_metrics"]

    threshold_table = _markdown_table(
        thresholds,
        [
            "fold",
            "unique_permitted_origin_count",
            "finite_rv21_count",
            "volatility_median_threshold",
            "finite_abs_return_count",
            "abs_return_q95_threshold",
        ],
    )
    failure_table = _markdown_table(
        failure,
        [
            "reporting_level",
            "model",
            "regime_dimension",
            "reportable_contrasts",
            "negative_degradation_count",
            "zero_within_tolerance_count",
            "positive_degradation_count",
            "lstm_seed_sign_disagreement_count",
        ],
    )
    skill_cross_fold = cross_fold.loc[
        cross_fold["degradation_metric"] == "mae_skill_degradation"
    ]
    cross_fold_table = _markdown_table(
        skill_cross_fold,
        [
            "model",
            "regime_dimension",
            "reportable_fold_count",
            "median",
            "q1",
            "q3",
            "complete_five_fold_summary",
            "summary_status",
        ],
    )
    disagreement_rows = disagreement.loc[disagreement["is_disagreement"]]
    disagreement_table = _markdown_table(
        disagreement_rows,
        [
            "model",
            "fold",
            "asset",
            "regime_dimension",
            "mae_skill_degradation",
            "da_degradation",
            "disagreement_type",
        ],
    )
    artifact_lines = "\n".join(
        f"- {name}: {path.relative_to(REPOSITORY_ROOT).as_posix()} "
        f"(SHA-256 {result.artifact_hashes[name]})"
        for name, path in result.artifact_paths.items()
    )
    unavailable = label_counts.loc[label_counts["regime_value"] == "unavailable"]
    low_n_asset = int(
        (degradation["contrast_status"] != "reportable").sum()
    )
    low_n_macro = int(
        (macro_degradation["contrast_status"] != "reportable").sum()
    )
    supported_lines: list[str] = []
    asset_failure = failure.loc[failure["reporting_level"] == "asset_fold"]
    for row in asset_failure.itertuples(index=False):
        supported_lines.append(
            f"- For {row.model} in the {row.regime_dimension} contrast, "
            f"{row.negative_degradation_count} of {row.reportable_contrasts} "
            "reportable development asset/fold contrasts had worse stressed-regime "
            "baseline-relative MAE skill."
        )

    sections: list[tuple[str, str]] = [
        (
            "Phase status",
            "**PHASE 2J DEVELOPMENT ANALYSIS COMPLETE, subject only to the recorded "
            "post-generation full-test line below.** All scientific artifacts were "
            "generated once from the frozen D1--D5 predictions and independently reloaded.",
        ),
        (
            "Frozen input integrity",
            "PASS. The processed dataset, three canonical prediction files, frozen "
            "contracts, and six historical result trees matched their preregistered "
            "SHA-256 identities before and after artifact generation.",
        ),
        (
            "No-training confirmation",
            "No model was fit, refit, optimized, or used to regenerate predictions. "
            "The Phase 2J source audit found zero training imports and zero training calls.",
        ),
        (
            "Development/F1 boundary",
            "Only D1--D5 test targets from 2020--2024 were analyzed. The loader checked "
            "target_date before accessing retained SPY values, no post-2024 row entered "
            "an analysis population or artifact, and F1/2025 performance was not evaluated.",
        ),
        (
            "SPY source construction",
            f"The processed source supplied {result.spy_verification['repeated_rows']} "
            "repeated target-asset rows and "
            f"{result.spy_verification['unique_origin_rows']} unique chronological SPY "
            "origin observations. Repeated SPY values and missingness agreed across all "
            "four target assets before deduplication; thresholds therefore gave each SPY "
            "date one vote.",
        ),
        (
            "Fold assignment",
            "Partition membership used target_date, while every regime value used SPY "
            "information through origin_date only. An origin in the previous calendar "
            "year remained in the next target year's test fold when its target_date required it.",
        ),
        (
            "Threshold methodology",
            "Each fold used its own training plus validation target-date population. "
            "Volatility used finite annualized population RV21 and its median; shock "
            "detection used the finite absolute SPY-return linear 95th percentile. "
            "Same-fold test, later-fold, pooled-test, and 2025 observations were excluded.",
        ),
        ("Fold threshold table", threshold_table),
        (
            "Regime labeling implementation",
            f"Exactly {len(tables['unique_regime_labels'])} unique fold/origin labels were "
            "created. Trend zero was negative, volatility equality was low, an RV jump "
            "ratio of 1.5 triggered transition, and return equality to q95 did not. "
            "Invalid jump denominators produced no infinity.",
        ),
        ("Trend counts", dimension_counts("trend")),
        ("Volatility counts", dimension_counts("volatility")),
        ("Transition counts", dimension_counts("transition")),
        (
            "Missing/unavailable counts",
            _markdown_table(
                unavailable,
                ["fold", "regime_dimension", "regime_value", "count"],
            ),
        ),
        (
            "Sample-size audit",
            f"The fixed rule n >= 30 was applied. All {len(baseline_metrics)} baseline, "
            f"{len(lightgbm_metrics)} LightGBM, and {len(lstm_summary)} primary LSTM "
            "cells were explicitly materialized, including n=0 cells. "
            f"{low_n_asset} of {len(degradation)} asset/fold primary contrasts and "
            f"{low_n_macro} of {len(macro_degradation)} macro contrasts were not headline-reportable.",
        ),
        (
            "Baseline regime metrics",
            f"The baseline table contains {len(baseline_metrics)}/120 cells with MAE, "
            "RMSE, directional accuracy, positive-direction balance, n, and status. "
            "Baseline skill against itself was not calculated.",
        ),
        (
            "LightGBM regime metrics",
            f"The LightGBM table contains {len(lightgbm_metrics)}/120 cells. Every learned "
            "cell uses the exact saved Phase 2C baseline keys for MAE Skill and DA difference.",
        ),
        (
            "LSTM per-seed regime metrics",
            f"The LSTM per-seed table contains {len(lstm_seed_metrics)}/360 cells for "
            "seeds 1729, 2718, and 31415. Predictions were never averaged.",
        ),
        (
            "LSTM seed-summary metrics",
            f"The LSTM summary contains {len(lstm_summary)}/120 primary cells. Each metric "
            "was summarized after per-seed calculation using mean, population standard "
            "deviation, minimum, and maximum.",
        ),
        ("Transition degradation results", _degradation_summary_text(degradation, "transition")),
        ("Volatility degradation results", _degradation_summary_text(degradation, "volatility")),
        ("Trend degradation results", _degradation_summary_text(degradation, "trend")),
        (
            "Asset/fold headline contrasts",
            f"{int((degradation['contrast_status'] == 'reportable').sum())} of "
            f"{len(degradation)} primary asset/fold contrasts met the two-sided n >= 30 "
            "rule. Non-reportable values remain descriptive and are not headline evidence.",
        ),
        (
            "Macro fold results",
            f"The macro table contains {len(macro)} rows, including baseline context, "
            "deterministic LightGBM, LSTM per-seed values, and LSTM seed summaries. "
            f"{int((macro_degradation['contrast_status'] == 'reportable').sum())} of "
            f"{len(macro_degradation)} primary macro contrasts had complete four-asset "
            "support on both sides.",
        ),
        ("Cross-fold descriptive results", cross_fold_table),
        ("Failure-pattern counts", failure_table),
        (
            "Directional/error disagreement",
            f"{len(disagreement_rows)} of {len(disagreement)} reportable asset/fold "
            "contrasts showed the preregistered XOR disagreement between worsening "
            "MAE Skill and worsening directional accuracy.\n\n" + disagreement_table,
        ),
        (
            "Low-n suppressed conclusions",
            f"{low_n_asset} asset/fold contrasts and {low_n_macro} macro contrasts were "
            "suppressed from headline interpretation by the frozen sample-size rule. "
            "Their counts and descriptive numerical values remain in the audit artifacts.",
        ),
        (
            "Supported development conclusions",
            "The following are descriptive associations, not causal claims:\n\n"
            + "\n".join(supported_lines),
        ),
        (
            "Conclusions not supported",
            "These results do not establish that a regime caused model failure, that any "
            "strategy is profitable, that the contrasts are statistically significant, "
            "that they generalize universally, or that F1/2025 confirms them.",
        ),
        (
            "Independent saved-file verification",
            "PASS. All sixteen primary CSV artifacts were reloaded from disk after writing. "
            "Thresholds, labels, joins, cell metrics, exact baseline references, seed "
            "summaries, degradations, macro layers, cross-fold summaries, failure counts, "
            "and disagreement classifications reproduced within 1e-12.",
        ),
        (
            "Historical artifact immutability",
            "PASS. Baseline, LightGBM, LSTM, Phase 2G combined, PatchTST pilot, and "
            "PatchTST authorization tree identities remained byte-identical. All seven "
            "frozen Markdown/YAML contracts retained their expected hashes.",
        ),
        ("Full test-suite result", full_test_result),
        (
            "Artifact locations/hashes",
            artifact_lines,
        ),
        (
            "Specification deviations",
            "None. The write-free dry run caught one representation-only implementation "
            "bug: equal target timestamps loaded at different NumPy resolutions failed a "
            "strict Series dtype comparison. The check was corrected to compare normalized "
            "nanosecond timestamp values before execution. No scientific rule, threshold, "
            "label, metric, or result changed.",
        ),
        (
            "Remaining risks",
            "Low-n transition cells can limit complete macro or five-fold conclusions. "
            "The three regime dimensions overlap and are descriptive views rather than "
            "independent causal mechanisms. Expanding fold histories make thresholds "
            "fold-specific by design, and LSTM seed dispersion can complicate a seed-mean result.",
        ),
        (
            "Recommended next phase",
            "**Phase 2K / Development Regime Results Audit and Final-Test Authorization "
            "Review.** Phase 2K was not started.",
        ),
    ]
    if len(sections) != 36:
        raise RegimeAnalysisError("Phase 2J report must contain exactly 36 sections.")
    lines = [
        "# Phase 2J — Development Regime Stress-Test Execution",
        "",
    ]
    for index, (title, body) in enumerate(sections, start=1):
        lines.extend([f"## {index}. {title}", "", body, ""])
    return "\n".join(lines).rstrip() + "\n"


def write_phase2j_report(
    result: Phase2JResult,
    *,
    full_test_result: str = "FINAL_TEST_RESULT_PENDING",
) -> Path:
    payload = build_phase2j_report(
        result,
        full_test_result=full_test_result,
    ).encode("utf-8")
    if REPORT_PATH.exists() and REPORT_PATH.read_bytes() != payload:
        raise RegimeAnalysisError(f"Refusing to overwrite existing {REPORT_PATH.name}.")
    if not REPORT_PATH.exists():
        REPORT_PATH.write_bytes(payload)
    return REPORT_PATH


__all__ = [
    "ASSETS",
    "CONTRASTS",
    "FOLDS",
    "LSTM_SEEDS",
    "MINIMUM_HEADLINE_ROWS",
    "NUMERIC_TOLERANCE",
    "Phase2JResult",
    "REGIME_VALUES",
    "RegimeAnalysisError",
    "assert_analysis_only_sources",
    "audit_prediction_keys",
    "build_label_counts",
    "build_phase2j_report",
    "classify_trend",
    "classify_volatility",
    "compute_asset_fold_degradation",
    "compute_baseline_regime_metrics",
    "compute_cross_fold_summary",
    "compute_directional_error_disagreement",
    "compute_failure_patterns",
    "compute_fold_thresholds",
    "compute_learned_metric_primitives",
    "compute_learned_regime_metrics",
    "compute_lstm_seed_summary",
    "compute_macro_degradation",
    "compute_macro_regime_metrics",
    "compute_metric_primitives",
    "complete_interrupted_phase2j_verification",
    "construct_unique_spy_view",
    "generate_regime_labels",
    "join_regime_labels",
    "load_development_spy_source",
    "match_exact_baseline_rows",
    "prediction_label_keys",
    "run_regime_analysis",
    "sample_size_status",
    "transition_components",
    "verify_cross_model_label_consistency",
    "write_phase2j_report",
]
