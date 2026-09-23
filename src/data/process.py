"""Deterministic construction of the locked supervised market dataset.

This module is deliberately separate from acquisition.  It performs no network
I/O and never evaluates a model.  The output keeps all mechanically available
target rows, including rows reserved for the guarded 2025 final fold.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import DEFAULT_DATA_CONFIG_PATH, DataConfig, load_data_config
from src.data.validation import (
    DataValidationError,
    normalize_canonical_data,
    validate_canonical_data,
)
from src.evaluation.walk_forward import DEFAULT_WALK_FORWARD_CONFIG_PATH, FoldRegistry
from src.features.market_features import (
    DEFAULT_ROLLING_WINDOWS,
    DEFAULT_SPY_TREND_WINDOW,
    build_market_features,
)
from src.features.targets import TARGET_COLUMNS, generate_targets


DEFAULT_PROCESSED_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "processed"
PROCESSED_SCHEMA_VERSION = 1
EXPERIMENT_SPEC_VERSION = "1.1"

LOCKED_FEATURE_COLUMNS: tuple[str, ...] = (
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
    "spy_observed",
)
CORE_NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column for column in LOCKED_FEATURE_COLUMNS if column != "spy_observed"
)
ELIGIBILITY_COLUMN = "core_evaluation_eligible"
PROCESSED_COLUMNS: tuple[str, ...] = (
    *TARGET_COLUMNS,
    *LOCKED_FEATURE_COLUMNS,
    ELIGIBILITY_COLUMN,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class ProcessedDatasetPaths:
    """Paths and digest for one immutable processed dataset."""

    data: Path
    manifest: Path
    sha256: str


def _expected_eligibility(frame: pd.DataFrame) -> np.ndarray:
    finite_features = np.isfinite(
        frame.loc[:, list(CORE_NUMERIC_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    ).all(axis=1)
    spy_available = frame["spy_observed"].to_numpy(dtype="int8") == 1
    return finite_features & spy_available


def _validate_processed_dataset(
    frame: pd.DataFrame,
    *,
    config: DataConfig,
) -> pd.DataFrame:
    """Normalize and strictly validate a processed dataset without dropping rows."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"frame must be a pandas DataFrame, got {type(frame).__name__}.")
    missing = set(PROCESSED_COLUMNS).difference(frame.columns)
    unknown = set(frame.columns).difference(PROCESSED_COLUMNS)
    if missing:
        raise DataValidationError(
            f"Processed dataset is missing columns: {sorted(missing)}."
        )
    if unknown:
        raise DataValidationError(
            f"Processed dataset has unsupported columns: {sorted(unknown)}."
        )

    normalized = frame.loc[:, list(PROCESSED_COLUMNS)].copy()
    normalized["asset"] = normalized["asset"].astype("string").str.upper()
    for column in ("origin_date", "target_date"):
        normalized[column] = pd.to_datetime(normalized[column], errors="raise")
        if normalized[column].dt.tz is not None:
            normalized[column] = normalized[column].dt.tz_convert(None)

    normalized = normalized.sort_values(
        ["asset", "origin_date", "target_date"],
        kind="mergesort",
        ignore_index=True,
    )
    if normalized.empty:
        raise DataValidationError("Processed dataset must not be empty.")
    observed_assets = tuple(
        asset for asset in config.assets if asset in set(normalized["asset"])
    )
    if observed_assets != config.assets or set(normalized["asset"]) != set(config.assets):
        raise DataValidationError(
            "Processed dataset must contain exactly the locked target stocks "
            f"{list(config.assets)} and no market-reference target rows."
        )
    if normalized.duplicated(
        ["asset", "origin_date", "target_date"], keep=False
    ).any():
        raise DataValidationError("Processed dataset has duplicate sample identities.")
    if (normalized["target_date"] <= normalized["origin_date"]).any():
        raise DataValidationError("Every target_date must be later than origin_date.")

    range_start = pd.Timestamp(config.start_date)
    range_end = pd.Timestamp(config.end_date)
    if not normalized["origin_date"].between(range_start, range_end).all():
        raise DataValidationError("Processed origin_date falls outside the locked range.")
    if not normalized["target_date"].between(range_start, range_end).all():
        raise DataValidationError("Processed target_date falls outside the locked range.")

    numeric_columns = (
        "target_log_return",
        "target_direction",
        *LOCKED_FEATURE_COLUMNS,
        ELIGIBILITY_COLUMN,
    )
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise")
    if not np.isfinite(normalized["target_log_return"].to_numpy(dtype="float64")).all():
        raise DataValidationError("Processed targets must contain finite log returns.")

    target_direction = normalized["target_direction"].to_numpy(dtype="float64")
    if not np.isin(target_direction, (0.0, 1.0)).all():
        raise DataValidationError("target_direction must contain only 0 and 1.")
    expected_direction = (
        normalized["target_log_return"].to_numpy(dtype="float64") > 0.0
    ).astype("int8")
    if not np.array_equal(target_direction.astype("int8"), expected_direction):
        raise DataValidationError(
            "target_direction must equal 1 iff target_log_return is strictly positive."
        )

    spy_observed = normalized["spy_observed"].to_numpy(dtype="float64")
    if not np.isin(spy_observed, (0.0, 1.0)).all():
        raise DataValidationError("spy_observed must contain only 0 and 1.")
    supplied_eligibility = normalized[ELIGIBILITY_COLUMN].to_numpy(dtype="float64")
    if not np.isin(supplied_eligibility, (0.0, 1.0)).all():
        raise DataValidationError(f"{ELIGIBILITY_COLUMN} must contain only 0 and 1.")

    normalized["target_direction"] = normalized["target_direction"].astype("int8")
    normalized["spy_observed"] = normalized["spy_observed"].astype("int8")
    normalized[ELIGIBILITY_COLUMN] = normalized[ELIGIBILITY_COLUMN].astype("int8")
    if not np.array_equal(
        normalized[ELIGIBILITY_COLUMN].to_numpy(dtype=bool),
        _expected_eligibility(normalized),
    ):
        raise DataValidationError(
            f"{ELIGIBILITY_COLUMN} does not match complete locked features and SPY availability."
        )
    return normalized


def build_supervised_dataset(
    canonical_data: pd.DataFrame,
    *,
    config: DataConfig | None = None,
) -> pd.DataFrame:
    """Build target-stock samples from the locked causal feature and target rules.

    Eligibility is independent of target values: a row is eligible exactly when
    every locked numeric input is finite and the same-date SPY observation exists.
    Ineligible warm-up and missing-reference rows remain in the output.
    """

    active_config = config or load_data_config()
    validate_canonical_data(
        canonical_data,
        config=active_config,
        require_sorted=True,
        require_all_assets=True,
    )
    # Stored canonical CSVs carry ISO date strings.  Normalize types only after
    # the strict ordering check so this builder does not silently repair input.
    canonical = normalize_canonical_data(
        canonical_data,
        config=active_config,
        require_all_assets=True,
    )

    target_rows = canonical.loc[
        canonical["asset"].isin(active_config.assets)
    ].copy()
    targets = generate_targets(target_rows, config=active_config)
    features = build_market_features(
        canonical,
        config=active_config,
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        spy_trend_window=DEFAULT_SPY_TREND_WINDOW,
        missing_spy="keep",
    )

    feature_columns = tuple(
        column
        for column in features.columns
        if column not in {"asset", "origin_date"}
    )
    if feature_columns != LOCKED_FEATURE_COLUMNS:
        raise DataValidationError(
            "Feature output differs from the feature set locked for processed schema 1."
        )

    merged = targets.merge(
        features,
        on=["asset", "origin_date"],
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise DataValidationError("At least one target row has no same-origin feature row.")
    merged = merged.drop(columns="_merge")
    merged[ELIGIBILITY_COLUMN] = _expected_eligibility(merged).astype("int8")

    return _validate_processed_dataset(merged, config=active_config)


def _iso_min(series: pd.Series) -> str | None:
    if series.empty:
        return None
    return pd.Timestamp(series.min()).date().isoformat()


def _iso_max(series: pd.Series) -> str | None:
    if series.empty:
        return None
    return pd.Timestamp(series.max()).date().isoformat()


def _eligibility_summary(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame[ELIGIBILITY_COLUMN].astype(bool)
    missing_spy = frame["spy_observed"].ne(1).to_numpy()
    complete_features = np.isfinite(
        frame.loc[:, list(CORE_NUMERIC_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    ).all(axis=1)
    feature_warmup = ~missing_spy & ~complete_features
    expected_eligible = ~missing_spy & complete_features
    if not np.array_equal(eligible.to_numpy(), expected_eligible):
        raise DataValidationError(
            f"{ELIGIBILITY_COLUMN} does not match the locked target-independent rule."
        )

    prefix_warmup_rows = 0
    for _, asset_rows in frame.groupby("asset", sort=True, observed=True):
        ordered = asset_rows.sort_values("origin_date", kind="mergesort")
        flags = ordered[ELIGIBILITY_COLUMN].to_numpy(dtype=bool)
        eligible_positions = np.flatnonzero(flags)
        prefix_warmup_rows += (
            int(eligible_positions[0]) if eligible_positions.size else len(flags)
        )

    row_count = int(len(frame))
    eligible_count = int(eligible.sum())
    ineligible_count = row_count - eligible_count
    missing_spy_count = int(missing_spy.sum())
    feature_warmup_count = int(feature_warmup.sum())
    eligible_rows = frame.loc[eligible]
    return {
        "row_count": row_count,
        "eligible_row_count": eligible_count,
        "ineligible_row_count": ineligible_count,
        "eligible_fraction": eligible_count / row_count if row_count else None,
        "feature_warmup_rows": feature_warmup_count,
        "missing_spy_alignment_rows": missing_spy_count,
        "other_ineligible_rows": (
            ineligible_count - feature_warmup_count - missing_spy_count
        ),
        "prefix_rows_before_first_eligible": prefix_warmup_rows,
        "origin_date_start": _iso_min(frame["origin_date"]),
        "origin_date_end": _iso_max(frame["origin_date"]),
        "target_date_start": _iso_min(frame["target_date"]),
        "target_date_end": _iso_max(frame["target_date"]),
        "first_eligible_origin_date": _iso_min(eligible_rows["origin_date"]),
        "first_eligible_target_date": _iso_min(eligible_rows["target_date"]),
    }


def _summary_with_assets(frame: pd.DataFrame, assets: tuple[str, ...]) -> dict[str, Any]:
    result = _eligibility_summary(frame)
    result["by_asset"] = {
        asset: _eligibility_summary(frame.loc[frame["asset"] == asset])
        for asset in assets
    }
    return result


def compute_eligibility_stats(
    dataset: pd.DataFrame,
    *,
    registry: FoldRegistry | None = None,
) -> dict[str, Any]:
    """Summarize deterministic core eligibility overall, by asset, and by D1-D5.

    Passing a fold registry adds train/validation/test summaries for development
    folds only.  The guarded F1 test interval is intentionally not summarized.
    """

    required = {
        "asset",
        "origin_date",
        "target_date",
        "spy_observed",
        ELIGIBILITY_COLUMN,
        *CORE_NUMERIC_FEATURE_COLUMNS,
    }
    missing = required.difference(dataset.columns)
    if missing:
        raise DataValidationError(
            f"Eligibility statistics require columns: {sorted(missing)}."
        )
    frame = dataset.loc[:, list(required)].copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise")
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
    frame[ELIGIBILITY_COLUMN] = pd.to_numeric(
        frame[ELIGIBILITY_COLUMN], errors="raise"
    ).astype("int8")
    assets = tuple(sorted(str(asset) for asset in frame["asset"].unique()))

    result: dict[str, Any] = {
        "definition": (
            "1 iff all locked numeric feature inputs are finite and a same-date "
            "SPY observation exists; targets and model performance are not inputs"
        ),
        "overall": _eligibility_summary(frame),
        "by_asset": {
            asset: _eligibility_summary(frame.loc[frame["asset"] == asset])
            for asset in assets
        },
    }
    if registry is not None:
        by_fold: dict[str, Any] = {}
        for fold_id in registry.default_development_folds:
            fold = registry.get(fold_id)
            partitions: dict[str, Any] = {}
            for partition_name in ("train", "validation", "test"):
                date_range = getattr(fold, partition_name)
                partition = frame.loc[date_range.contains(frame["target_date"])]
                partitions[partition_name] = _summary_with_assets(partition, assets)
            by_fold[fold_id] = partitions
        result["by_development_fold"] = by_fold
    return result


def _processed_csv_bytes(frame: pd.DataFrame) -> bytes:
    serializable = frame.loc[:, list(PROCESSED_COLUMNS)].copy()
    for column in ("origin_date", "target_date"):
        serializable[column] = pd.to_datetime(serializable[column]).dt.strftime(
            "%Y-%m-%d"
        )
    text = serializable.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return text.encode("utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_provenance() -> dict[str, Any]:
    source_root = Path(__file__).parents[1]
    source_paths = {
        "src/data/process.py": Path(__file__),
        "src/features/targets.py": source_root / "features" / "targets.py",
        "src/features/market_features.py": source_root / "features" / "market_features.py",
    }
    configuration_paths = {
        "configs/data.yaml": DEFAULT_DATA_CONFIG_PATH,
        "configs/walk_forward.yaml": DEFAULT_WALK_FORWARD_CONFIG_PATH,
    }
    return {
        "revision_method": "sha256_by_source_and_configuration_file",
        "source_files": {
            name: _file_sha256(path) for name, path in source_paths.items()
        },
        "configuration_files": {
            name: _file_sha256(path) for name, path in configuration_paths.items()
        },
    }


def _definitions() -> dict[str, Any]:
    return {
        "targets": {
            "scope": "asset-local target stocks only; SPY is never a prediction target",
            "horizon": "next observed session for the same asset",
            "target_log_return": "ln(close[target_date] / close[origin_date])",
            "target_direction": "1 iff target_log_return > 0; otherwise 0",
        },
        "features": {
            "information_cutoff": "origin-date close; no later values",
            "rolling_windows": list(DEFAULT_ROLLING_WINDOWS),
            "spy_trend_window": DEFAULT_SPY_TREND_WINDOW,
            "spy_alignment": "same origin_date; never positional or filled",
            "columns": list(LOCKED_FEATURE_COLUMNS),
        },
        "eligibility": {
            "column": ELIGIBILITY_COLUMN,
            "rule": (
                "1 iff all locked numeric feature inputs are finite and "
                "spy_observed == 1"
            ),
            "target_independent": True,
        },
    }


def write_immutable_processed_dataset(
    dataset: pd.DataFrame,
    *,
    raw_source_sha256: str,
    raw_snapshot_file: str | Path | None = None,
    output_directory: str | Path = DEFAULT_PROCESSED_DIRECTORY,
    config: DataConfig | None = None,
    registry: FoldRegistry | None = None,
    created_at: datetime | None = None,
) -> ProcessedDatasetPaths:
    """Write a content-addressed processed CSV and provenance manifest once."""

    if not _SHA256_PATTERN.fullmatch(raw_source_sha256):
        raise ValueError("raw_source_sha256 must be exactly 64 hexadecimal characters.")
    active_config = config or load_data_config()
    normalized = _validate_processed_dataset(dataset, config=active_config)
    payload = _processed_csv_bytes(normalized)
    digest = hashlib.sha256(payload).hexdigest()

    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("created_at must be timezone-aware.")
    timestamp = timestamp.astimezone(UTC)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"supervised_v{PROCESSED_SCHEMA_VERSION}_{digest}"
    data_path = output_path / f"{stem}.csv"
    manifest_path = output_path / f"{stem}.manifest.json"
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Immutable processed artifact already exists; refusing to overwrite "
            f"{data_path} or {manifest_path}."
        )

    row_count_by_asset = {
        asset: int(len(asset_rows))
        for asset, asset_rows in normalized.groupby("asset", sort=True, observed=True)
    }
    date_range_by_asset = {
        asset: {
            "origin_start": _iso_min(asset_rows["origin_date"]),
            "origin_end": _iso_max(asset_rows["origin_date"]),
            "target_start": _iso_min(asset_rows["target_date"]),
            "target_end": _iso_max(asset_rows["target_date"]),
        }
        for asset, asset_rows in normalized.groupby("asset", sort=True, observed=True)
    }
    manifest = {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "code_version": f"processed_schema_v{PROCESSED_SCHEMA_VERSION}",
        "code_provenance": _code_provenance(),
        "experiment_spec_version": EXPERIMENT_SPEC_VERSION,
        "created_timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "assets": list(active_config.assets),
        "processed_file": data_path.name,
        "file_sha256": digest,
        "source": {
            "raw_file": Path(raw_snapshot_file).name if raw_snapshot_file else None,
            "raw_file_sha256": raw_source_sha256.lower(),
        },
        "columns": list(PROCESSED_COLUMNS),
        "definitions": _definitions(),
        "counts": {
            "row_count": int(len(normalized)),
            "row_count_by_asset": row_count_by_asset,
        },
        "date_ranges": {
            "origin_start": _iso_min(normalized["origin_date"]),
            "origin_end": _iso_max(normalized["origin_date"]),
            "target_start": _iso_min(normalized["target_date"]),
            "target_end": _iso_max(normalized["target_date"]),
            "by_asset": date_range_by_asset,
        },
        "eligibility": compute_eligibility_stats(normalized, registry=registry),
    }

    created_data = False
    created_manifest = False
    try:
        with data_path.open("xb") as handle:
            handle.write(payload)
        created_data = True
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        created_manifest = True
    except Exception:
        # Remove only files created by this invocation; existing files are rejected above.
        if created_manifest:
            manifest_path.unlink(missing_ok=True)
        if created_data:
            data_path.unlink(missing_ok=True)
        raise

    return ProcessedDatasetPaths(data=data_path, manifest=manifest_path, sha256=digest)


def create_processed_dataset(
    canonical_data: pd.DataFrame,
    *,
    raw_source_sha256: str,
    raw_snapshot_file: str | Path | None = None,
    output_directory: str | Path = DEFAULT_PROCESSED_DIRECTORY,
    config: DataConfig | None = None,
    registry: FoldRegistry | None = None,
    created_at: datetime | None = None,
) -> ProcessedDatasetPaths:
    """Build and persist one immutable content-addressed supervised dataset."""

    active_config = config or load_data_config()
    dataset = build_supervised_dataset(canonical_data, config=active_config)
    return write_immutable_processed_dataset(
        dataset,
        raw_source_sha256=raw_source_sha256,
        raw_snapshot_file=raw_snapshot_file,
        output_directory=output_directory,
        config=active_config,
        registry=registry,
        created_at=created_at,
    )
