"""Development-only execution and immutable artifacts for locked baselines."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.schema import DEFAULT_DATA_CONFIG_PATH, TARGET_ASSETS
from src.evaluation.walk_forward import (
    DEFAULT_WALK_FORWARD_CONFIG_PATH,
    DEVELOPMENT_FOLD_IDS,
    FoldRegistry,
    get_partition,
    load_fold_registry,
)
from src.models.baselines import (
    COMBINED_BASELINE_ID,
    DIRECTION_BASELINE_ID,
    REGRESSION_BASELINE_ID,
    direction_persistence_prediction,
    zero_return_prediction,
)


SPEC_VERSION = "1.1"
BASELINE_MODEL_CONFIG_ID = "zero_return__direction_persistence_v1"
BASELINE_SEED = "not_applicable"
REGIME_LABEL_PLACEHOLDER = "not_labeled_phase2c"
REGIME_COLUMNS: tuple[str, ...] = (
    "trend_regime",
    "volatility_regime",
    "transition_regime",
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
    *REGIME_COLUMNS,
)

PER_ASSET_METRIC_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
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
    "skill_score_vs_self",
    "skill_interpretation",
)

MACRO_METRIC_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "fold",
    "n_assets",
    "n_observations",
    "mae",
    "rmse",
    "directional_accuracy",
    "actual_positive_rate",
    "actual_nonpositive_rate",
    "macro_weighting",
    "skill_score_vs_self",
    "skill_interpretation",
)

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "asset",
    "origin_date",
    "target_date",
    "target_log_return",
    "target_direction",
    "asset_log_return_1d",
    "core_evaluation_eligible",
)

_SELF_SKILL_INTERPRETATION = "zero_by_definition_not_meaningful"


@dataclass(frozen=True)
class BaselineArtifacts:
    """In-memory outputs plus the immutable paths and hashes written to disk."""

    predictions: pd.DataFrame
    per_asset_metrics: pd.DataFrame
    macro_metrics: pd.DataFrame
    prediction_path: Path
    per_asset_metrics_path: Path
    macro_metrics_path: Path
    manifest_path: Path
    artifact_hashes: Mapping[str, str]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    stream = io.StringIO(newline="")
    frame.to_csv(
        stream,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return stream.getvalue().encode("utf-8")


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for distribution in ("PyYAML", "yfinance", "exchange-calendars"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to overwrite immutable artifact {path}.")
        return
    path.write_bytes(payload)


def _validated_eligibility(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("core_evaluation_eligible must not contain missing values.")
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    numeric = pd.to_numeric(values, errors="raise")
    if not numeric.isin([0, 1]).all():
        raise ValueError("core_evaluation_eligible must contain only boolean or 0/1 values.")
    return numeric.astype(bool)


def _validate_input(samples: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(samples, pd.DataFrame):
        raise TypeError(f"samples must be a pandas DataFrame, got {type(samples).__name__}.")
    missing = sorted(set(REQUIRED_INPUT_COLUMNS).difference(samples.columns))
    if missing:
        raise ValueError(f"Processed samples are missing required columns: {missing}.")

    result = samples.copy()
    result["asset"] = result["asset"].astype("string")
    unknown_assets = sorted(set(result["asset"].dropna()).difference(TARGET_ASSETS))
    if unknown_assets:
        raise ValueError(f"Processed samples contain non-target assets: {unknown_assets}.")
    for column in ("origin_date", "target_date"):
        result[column] = pd.to_datetime(result[column], errors="raise")
        if result[column].isna().any():
            raise ValueError(f"{column} must not contain missing values.")
    if (result["target_date"] <= result["origin_date"]).any():
        raise ValueError("Every target_date must be strictly later than origin_date.")

    for column in ("target_log_return", "asset_log_return_1d"):
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    if not np.isfinite(result["target_log_return"].to_numpy()).all():
        raise ValueError("target_log_return must contain only finite values.")
    target_direction = pd.to_numeric(result["target_direction"], errors="raise")
    if not target_direction.isin([0, 1]).all():
        raise ValueError("target_direction must contain only binary 0/1 values.")
    result["target_direction"] = target_direction.astype("int8")
    expected_direction = (result["target_log_return"] > 0).astype("int8")
    if not result["target_direction"].equals(expected_direction):
        raise ValueError("target_direction is inconsistent with target_log_return.")
    result["core_evaluation_eligible"] = _validated_eligibility(
        result["core_evaluation_eligible"]
    )

    identity = ["asset", "origin_date", "target_date"]
    if result.duplicated(identity, keep=False).any():
        raise ValueError("Processed samples contain duplicate supervised-sample identities.")
    return result


def generate_development_predictions(
    samples: pd.DataFrame,
    *,
    run_id: str,
    data_version: str,
    registry: FoldRegistry | None = None,
    expected_assets: Sequence[str] = TARGET_ASSETS,
) -> pd.DataFrame:
    """Generate canonical predictions on eligible D1--D5 test rows only."""

    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")
    if not isinstance(data_version, str) or not data_version.strip():
        raise ValueError("data_version must be a non-empty string.")
    active_registry = registry or load_fold_registry()
    if active_registry.default_fold_ids() != DEVELOPMENT_FOLD_IDS:
        raise ValueError("Baseline execution requires the frozen D1--D5 development folds.")
    normalized = _validate_input(samples)
    required_assets = tuple(expected_assets)
    if required_assets != TARGET_ASSETS:
        raise ValueError(f"expected_assets must be exactly {TARGET_ASSETS}.")

    outputs: list[pd.DataFrame] = []
    for fold_id in DEVELOPMENT_FOLD_IDS:
        fold_test = get_partition(normalized, fold_id, "test", registry=active_registry)
        eligible = fold_test.loc[fold_test["core_evaluation_eligible"]].copy()
        observed_assets = set(eligible["asset"].dropna())
        if observed_assets != set(required_assets):
            missing = sorted(set(required_assets).difference(observed_assets))
            extra = sorted(observed_assets.difference(required_assets))
            raise ValueError(
                f"Fold {fold_id} eligible test rows do not form the required asset grid; "
                f"missing={missing}, extra={extra}."
            )

        output = pd.DataFrame(index=eligible.index)
        output["run_id"] = run_id
        output["spec_version"] = SPEC_VERSION
        output["data_version"] = data_version
        output["model"] = COMBINED_BASELINE_ID
        output["model_config_id"] = BASELINE_MODEL_CONFIG_ID
        output["seed"] = BASELINE_SEED
        output["fold"] = fold_id
        output["partition"] = "test"
        output["asset"] = eligible["asset"].astype("string")
        output["origin_date"] = eligible["origin_date"]
        output["target_date"] = eligible["target_date"]
        output["actual_log_return"] = eligible["target_log_return"].astype("float64")
        output["predicted_log_return"] = zero_return_prediction(eligible.index)
        output["actual_direction"] = eligible["target_direction"].astype("int8")
        output["predicted_direction"] = direction_persistence_prediction(
            eligible["asset_log_return_1d"]
        )
        for regime_column in REGIME_COLUMNS:
            output[regime_column] = REGIME_LABEL_PLACEHOLDER
        outputs.append(output.loc[:, list(PREDICTION_COLUMNS)])

    predictions = pd.concat(outputs, ignore_index=True).sort_values(
        ["fold", "asset", "target_date", "origin_date"],
        kind="mergesort",
        ignore_index=True,
    )
    if set(predictions["fold"]) != set(DEVELOPMENT_FOLD_IDS):
        raise AssertionError("Baseline output must contain exactly D1--D5.")
    target_years = predictions["target_date"].dt.year
    if (target_years >= 2025).any() or target_years.min() < 2020:
        raise AssertionError("Baseline output escaped the locked 2020--2024 development tests.")
    return predictions


def compute_per_asset_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics independently for every development fold and asset."""

    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"Prediction columns must be exactly {PREDICTION_COLUMNS}.")
    if predictions.empty:
        raise ValueError("Predictions must not be empty.")
    if predictions.duplicated(["fold", "asset", "origin_date", "target_date"]).any():
        raise ValueError("Predictions contain duplicate fold/sample identities.")
    if not set(predictions["fold"]).issubset(DEVELOPMENT_FOLD_IDS):
        raise ValueError("Metrics may only be computed for D1--D5 predictions.")
    if set(predictions["partition"]) != {"test"}:
        raise ValueError("Development metrics may only use test-partition predictions.")
    target_dates = pd.to_datetime(predictions["target_date"], errors="raise")
    if (target_dates.dt.year >= 2025).any():
        raise ValueError("Final-test/2025 predictions are forbidden in development metrics.")

    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(["run_id", "model", "fold", "asset"], sort=True)
    for (run_id, model, fold, asset), frame in grouped:
        actual = frame["actual_log_return"].to_numpy(dtype="float64")
        predicted = frame["predicted_log_return"].to_numpy(dtype="float64")
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError("Regression values must be finite before metric computation.")
        actual_direction = frame["actual_direction"].to_numpy(dtype="int8")
        predicted_direction = frame["predicted_direction"].to_numpy(dtype="int8")
        if not np.isin(actual_direction, [0, 1]).all() or not np.isin(
            predicted_direction, [0, 1]
        ).all():
            raise ValueError("Direction values must be binary before metric computation.")
        errors = predicted - actual
        positive_count = int(actual_direction.sum())
        n = len(frame)
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
                "skill_interpretation": _SELF_SKILL_INTERPRETATION,
            }
        )
    result = pd.DataFrame(rows, columns=PER_ASSET_METRIC_COLUMNS)
    return result.sort_values(["fold", "asset"], kind="mergesort", ignore_index=True)


def compute_macro_metrics(per_asset_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-weight fold macro metrics across the four target assets."""

    if tuple(per_asset_metrics.columns) != PER_ASSET_METRIC_COLUMNS:
        raise ValueError(f"Per-asset metric columns must be exactly {PER_ASSET_METRIC_COLUMNS}.")
    rows: list[dict[str, Any]] = []
    grouped = per_asset_metrics.groupby(["run_id", "model", "fold"], sort=True)
    for (run_id, model, fold), frame in grouped:
        if set(frame["asset"]) != set(TARGET_ASSETS) or len(frame) != len(TARGET_ASSETS):
            raise ValueError(f"Fold {fold} must have exactly one metric row per target asset.")
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
                "skill_interpretation": _SELF_SKILL_INTERPRETATION,
            }
        )
    result = pd.DataFrame(rows, columns=MACRO_METRIC_COLUMNS)
    return result.sort_values(["fold"], kind="mergesort", ignore_index=True)


def run_and_save_development_baselines(
    samples: pd.DataFrame,
    *,
    output_root: str | Path,
    run_id: str,
    source_snapshot_sha256: str,
    processed_dataset_sha256: str,
    created_at: datetime,
    provenance: Mapping[str, Any] | None = None,
    registry: FoldRegistry | None = None,
) -> BaselineArtifacts:
    """Run D1--D5 baselines and persist immutable, content-addressed artifacts."""

    for name, digest in (
        ("source_snapshot_sha256", source_snapshot_sha256),
        ("processed_dataset_sha256", processed_dataset_sha256),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} must be a 64-character SHA-256 hex digest.")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"{name} must be a hexadecimal SHA-256 digest.") from exc
    if created_at.tzinfo is None:
        raise ValueError("created_at must be timezone-aware.")
    created_timestamp_utc = created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")

    active_registry = registry or load_fold_registry()
    predictions = generate_development_predictions(
        samples,
        run_id=run_id,
        data_version=processed_dataset_sha256.lower(),
        registry=active_registry,
    )
    per_asset = compute_per_asset_metrics(predictions)
    macro = compute_macro_metrics(per_asset)

    root = Path(output_root)
    payloads = {
        "predictions": _canonical_csv_bytes(predictions),
        "per_asset_metrics": _canonical_csv_bytes(per_asset),
        "macro_metrics": _canonical_csv_bytes(macro),
    }
    hashes = {name: _sha256(payload) for name, payload in payloads.items()}
    paths = {
        "predictions": root / "predictions" / f"baseline_predictions_{hashes['predictions']}.csv",
        "per_asset_metrics": root
        / "metrics"
        / f"baseline_metrics_by_asset_{hashes['per_asset_metrics']}.csv",
        "macro_metrics": root
        / "metrics"
        / f"baseline_metrics_macro_{hashes['macro_metrics']}.csv",
    }
    for name, path in paths.items():
        _write_immutable(path, payloads[name])

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_timestamp_utc": created_timestamp_utc,
        "experiment_spec_version": SPEC_VERSION,
        "configuration": {
            "data_config": {
                "path": "configs/data.yaml",
                "sha256": _sha256(DEFAULT_DATA_CONFIG_PATH.read_bytes()),
            },
            "walk_forward_config": {
                "path": "configs/walk_forward.yaml",
                "sha256": _sha256(DEFAULT_WALK_FORWARD_CONFIG_PATH.read_bytes()),
            },
        },
        "package_versions": _package_versions(),
        "evaluation": {
            "folds": list(DEVELOPMENT_FOLD_IDS),
            "partition": "test",
            "target_years": [2020, 2021, 2022, 2023, 2024],
            "eligibility_filter": "core_evaluation_eligible == true",
            "final_fold_excluded": True,
            "macro_weighting": "equal_weight_across_assets",
        },
        "prediction_contract": {
            "columns": list(PREDICTION_COLUMNS),
            "regime_label_status": "deferred_placeholder_only_no_regime_analysis",
            "regime_label_placeholder": REGIME_LABEL_PLACEHOLDER,
        },
        "models": {
            "combined_prediction_record_id": COMBINED_BASELINE_ID,
            "regression": {
                "id": REGRESSION_BASELINE_ID,
                "definition": "predicted next-day log return is always 0.0",
            },
            "direction": {
                "id": DIRECTION_BASELINE_ID,
                "definition": (
                    "predicted next-day direction is 1 iff the current asset log return "
                    "is strictly positive; otherwise 0"
                ),
            },
            "skill_score_vs_self": {
                "value": 0.0,
                "interpretation": _SELF_SKILL_INTERPRETATION,
            },
        },
        "source_revision": {
            "method": "sha256_by_source_file",
            "files": {
                "src/evaluation/baseline_evaluation.py": _sha256(Path(__file__).read_bytes()),
                "src/evaluation/walk_forward.py": _sha256(
                    (Path(__file__).with_name("walk_forward.py")).read_bytes()
                ),
                "src/evaluation/run_baselines.py": _sha256(
                    (Path(__file__).with_name("run_baselines.py")).read_bytes()
                ),
                "src/models/baselines.py": _sha256(
                    (Path(__file__).parents[1] / "models" / "baselines.py").read_bytes()
                ),
            },
        },
        "environment_lock": {
            "path": "pyproject.toml",
            "sha256": _sha256(DEFAULT_DATA_CONFIG_PATH.parents[1].joinpath("pyproject.toml").read_bytes()),
        },
        "fitted_preprocessing_artifact_identifiers": [],
        "model_configuration": {
            "model_config_id": BASELINE_MODEL_CONFIG_ID,
            "seed": BASELINE_SEED,
            "fitted_parameters": False,
        },
        "training_validation_date_bounds": {
            fold_id: {
                "train": {
                    "start": active_registry.get(fold_id).train.start.isoformat(),
                    "end": active_registry.get(fold_id).train.end.isoformat(),
                },
                "validation": {
                    "start": active_registry.get(fold_id).validation.start.isoformat(),
                    "end": active_registry.get(fold_id).validation.end.isoformat(),
                },
                "test": {
                    "start": active_registry.get(fold_id).test.start.isoformat(),
                    "end": active_registry.get(fold_id).test.end.isoformat(),
                },
            }
            for fold_id in DEVELOPMENT_FOLD_IDS
        },
        "runtime": {"status": "completed", "exceptions": []},
        "input_provenance": {
            **dict(provenance or {}),
            "source_snapshot_sha256": source_snapshot_sha256.lower(),
            "processed_dataset_sha256": processed_dataset_sha256.lower(),
        },
        "counts": {
            "prediction_rows": len(predictions),
            "per_asset_metric_rows": len(per_asset),
            "macro_metric_rows": len(macro),
        },
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashes[name],
            }
            for name, path in paths.items()
        },
    }
    manifest_payload = _canonical_json_bytes(manifest)
    manifest_hash = _sha256(manifest_payload)
    manifest_path = root / f"run_manifest_{manifest_hash}.json"
    _write_immutable(manifest_path, manifest_payload)
    all_hashes = {**hashes, "run_manifest": manifest_hash}

    return BaselineArtifacts(
        predictions=predictions,
        per_asset_metrics=per_asset,
        macro_metrics=macro,
        prediction_path=paths["predictions"],
        per_asset_metrics_path=paths["per_asset_metrics"],
        macro_metrics_path=paths["macro_metrics"],
        manifest_path=manifest_path,
        artifact_hashes=all_hashes,
    )
