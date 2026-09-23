"""Canonical market-data normalization and strict validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from src.data.schema import (
    CANONICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PRICE_COLUMNS,
    DataConfig,
    load_data_config,
)


class DataValidationError(ValueError):
    """Raised when market data violates the immutable data contract."""


def _require_dataframe(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(
            f"Expected a pandas DataFrame, got {type(frame).__name__}."
        )


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required canonical columns: {missing}.")


def _parse_dates(series: pd.Series, timezone: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Unparseable date value: {exc}") from exc
    if parsed.isna().any():
        raise DataValidationError("Date column contains missing values.")

    try:
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            parsed = parsed.dt.tz_convert(timezone).dt.tz_localize(None)
        elif not is_datetime64_any_dtype(parsed.dtype):
            parsed = pd.to_datetime(parsed, errors="raise", utc=True)
            parsed = parsed.dt.tz_convert(timezone).dt.tz_localize(None)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DataValidationError(
            "Dates must use one parseable, consistent timezone/date representation."
        ) from exc
    return parsed


def normalize_canonical_data(
    frame: pd.DataFrame,
    *,
    config: DataConfig | None = None,
    require_all_assets: bool = False,
) -> pd.DataFrame:
    """Explicitly normalize types and ordering, then validate without repairing data.

    Extra provider columns are intentionally excluded from the canonical raw schema.
    Invalid values and duplicate observations are rejected rather than fixed.
    """

    _require_dataframe(frame)
    _require_columns(frame, CANONICAL_COLUMNS)
    active_config = config or load_data_config()

    normalized = frame.loc[:, list(CANONICAL_COLUMNS)].copy()
    normalized["asset"] = normalized["asset"].astype("string").str.strip().str.upper()
    if normalized["asset"].isna().any() or (normalized["asset"] == "").any():
        raise DataValidationError("Asset column contains missing or empty ticker values.")

    parsed_dates = _parse_dates(normalized["date"], active_config.timezone)
    normalized["date"] = parsed_dates.dt.normalize()

    for column in NUMERIC_COLUMNS:
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(
                "float64"
            )
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Column {column!r} must be numeric: {exc}") from exc

    normalized = normalized.sort_values(
        ["asset", "date"], kind="mergesort", ignore_index=True
    )
    validate_canonical_data(
        normalized,
        config=active_config,
        require_sorted=True,
        require_all_assets=require_all_assets,
    )
    return normalized


def validate_canonical_data(
    frame: pd.DataFrame,
    *,
    config: DataConfig | None = None,
    require_sorted: bool = True,
    require_all_assets: bool = False,
) -> None:
    """Validate canonical daily market data and raise on every contract violation."""

    _require_dataframe(frame)
    _require_columns(frame, CANONICAL_COLUMNS)
    active_config = config or load_data_config()

    assets = frame["asset"].astype("string").str.strip().str.upper()
    if assets.isna().any() or (assets == "").any():
        raise DataValidationError("Asset column contains missing or empty ticker values.")

    observed_assets = set(assets.astype(str))
    allowed_assets = set(active_config.all_assets)
    unknown = observed_assets.difference(allowed_assets)
    if unknown:
        raise DataValidationError(
            f"Assets outside the locked universe were found: {sorted(unknown)}."
        )
    if require_all_assets:
        missing_assets = allowed_assets.difference(observed_assets)
        if missing_assets:
            raise DataValidationError(
                f"Complete snapshot is missing locked assets: {sorted(missing_assets)}."
            )

    parsed_dates = _parse_dates(frame["date"], active_config.timezone)
    normalized_dates = parsed_dates.dt.normalize()
    if not parsed_dates.equals(normalized_dates):
        raise DataValidationError("Canonical dates must be timezone-free session dates at midnight.")

    start = pd.Timestamp(active_config.start_date)
    end = pd.Timestamp(active_config.end_date)
    before_start = normalized_dates < start
    after_end = normalized_dates > end
    if before_start.any() or after_end.any():
        examples = frame.loc[before_start | after_end, ["asset", "date"]].head(5)
        raise DataValidationError(
            "Dates fall outside the locked range "
            f"{start.date()} through {end.date()}: {examples.to_dict('records')}."
        )
    weekend = normalized_dates.dt.dayofweek >= 5
    if weekend.any():
        examples = frame.loc[weekend, ["asset", "date"]].head(5)
        raise DataValidationError(
            f"Weekend rows are not observed trading sessions: {examples.to_dict('records')}."
        )

    keys = pd.DataFrame({"asset": assets.astype(str), "date": normalized_dates})
    duplicate_mask = keys.duplicated(["asset", "date"], keep=False)
    if duplicate_mask.any():
        examples = keys.loc[duplicate_mask].head(5).to_dict("records")
        raise DataValidationError(f"Duplicate (asset, date) rows found: {examples}.")

    if require_sorted:
        expected_order = keys.sort_values(
            ["asset", "date"], kind="mergesort", ignore_index=True
        )
        if not keys.reset_index(drop=True).equals(expected_order):
            raise DataValidationError("Rows must be sorted ascending by (asset, date).")

    for column in NUMERIC_COLUMNS:
        if not is_numeric_dtype(frame[column]):
            raise DataValidationError(f"Column {column!r} must have a numeric dtype.")
        values = frame[column].to_numpy(dtype="float64", copy=False)
        if not np.isfinite(values).all():
            bad_rows = np.flatnonzero(~np.isfinite(values))[:5].tolist()
            raise DataValidationError(
                f"Column {column!r} contains missing or non-finite values at rows {bad_rows}."
            )

    for column in PRICE_COLUMNS:
        if (frame[column] <= 0).any():
            raise DataValidationError(f"Column {column!r} contains non-positive prices.")
    if (frame["volume"] < 0).any():
        raise DataValidationError("Volume must be non-negative.")

    relationships = {
        "high must be greater than or equal to open": frame["high"] >= frame["open"],
        "high must be greater than or equal to close": frame["high"] >= frame["close"],
        "low must be less than or equal to open": frame["low"] <= frame["open"],
        "low must be less than or equal to close": frame["low"] <= frame["close"],
        "high must be greater than or equal to low": frame["high"] >= frame["low"],
    }
    for message, valid_mask in relationships.items():
        if not valid_mask.all():
            rows = np.flatnonzero(~valid_mask.to_numpy())[:5].tolist()
            raise DataValidationError(f"Invalid OHLC relationship ({message}) at rows {rows}.")


def validate_snapshot_file(path: str | Path, *, config: DataConfig | None = None) -> None:
    """Validate a stored canonical CSV without changing it."""

    snapshot_path = Path(path)
    try:
        frame = pd.read_csv(snapshot_path)
    except (OSError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"Unable to read snapshot {snapshot_path}: {exc}") from exc
    validate_canonical_data(
        frame,
        config=config,
        require_sorted=True,
        require_all_assets=True,
    )
