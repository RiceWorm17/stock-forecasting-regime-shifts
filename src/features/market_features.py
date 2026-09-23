"""Minimal causal market features for validating the Phase 2B foundation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from src.data.schema import MARKET_REFERENCES, TARGET_ASSETS, DataConfig
from src.data.validation import DataValidationError, validate_canonical_data


DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (5, 21)
DEFAULT_SPY_TREND_WINDOW = 63


def _validate_windows(windows: Sequence[int], trend_window: int) -> tuple[int, ...]:
    normalized = tuple(int(window) for window in windows)
    if not normalized or any(window < 2 for window in normalized):
        raise ValueError("Rolling windows must contain integers of at least 2.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Rolling windows must be unique.")
    if trend_window < 2:
        raise ValueError("SPY trend window must be at least 2.")
    return normalized


def _rolling_by_asset(
    frame: pd.DataFrame,
    value_column: str,
    window: int,
    statistic: Literal["mean", "std"],
) -> pd.Series:
    grouped = frame.groupby("asset", sort=False, observed=True)[value_column]
    if statistic == "mean":
        values = grouped.transform(
            lambda series: series.rolling(window, min_periods=window).mean()
        )
    else:
        values = grouped.transform(
            lambda series: series.rolling(window, min_periods=window).std(ddof=0)
        )
    return values.astype("float64")


def build_market_features(
    canonical_data: pd.DataFrame,
    *,
    config: DataConfig | None = None,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    spy_trend_window: int = DEFAULT_SPY_TREND_WINDOW,
    missing_spy: Literal["raise", "keep"] = "raise",
) -> pd.DataFrame:
    """Build causal target-asset and SPY features keyed by asset/date.

    Rolling windows include information through the origin close and never use
    centered windows. A missing SPY key is either rejected or retained as an
    explicit missing observation; it is never positionally joined or backfilled.
    """

    validate_canonical_data(
        canonical_data,
        config=config,
        require_sorted=True,
        require_all_assets=False,
    )
    windows = _validate_windows(rolling_windows, spy_trend_window)
    if missing_spy not in {"raise", "keep"}:
        raise ValueError("missing_spy must be either 'raise' or 'keep'.")

    configured_targets = config.assets if config is not None else TARGET_ASSETS
    configured_references = (
        config.market_reference if config is not None else MARKET_REFERENCES
    )
    if len(configured_references) != 1:
        raise DataValidationError("Exactly one market reference is required.")
    spy_asset = configured_references[0]

    target_frame = canonical_data.loc[
        canonical_data["asset"].isin(configured_targets)
    ].copy()
    spy_frame = canonical_data.loc[canonical_data["asset"] == spy_asset].copy()
    if target_frame.empty:
        raise DataValidationError("No locked target-stock rows are available.")
    if spy_frame.empty:
        raise DataValidationError(f"No {spy_asset} market-reference rows are available.")

    target_group = target_frame.groupby("asset", sort=False, observed=True)
    target_frame["asset_log_return_1d"] = target_group["close"].transform(
        lambda series: np.log(series / series.shift(1))
    )
    target_frame["asset_log_return_lag_1"] = target_frame.groupby(
        "asset", sort=False, observed=True
    )["asset_log_return_1d"].shift(1)

    for window in windows:
        target_frame[f"asset_return_mean_{window}"] = _rolling_by_asset(
            target_frame, "asset_log_return_1d", window, "mean"
        )
        target_frame[f"asset_return_std_{window}"] = _rolling_by_asset(
            target_frame, "asset_log_return_1d", window, "std"
        )
        target_frame[f"asset_volume_mean_{window}"] = _rolling_by_asset(
            target_frame, "volume", window, "mean"
        )
        target_frame[f"asset_volume_std_{window}"] = _rolling_by_asset(
            target_frame, "volume", window, "std"
        )

    spy_frame = spy_frame.sort_values("date", kind="mergesort").copy()
    spy_frame["spy_log_return_1d"] = np.log(
        spy_frame["close"] / spy_frame["close"].shift(1)
    )
    spy_frame["spy_log_return_lag_1"] = spy_frame["spy_log_return_1d"].shift(1)
    for window in windows:
        spy_frame[f"spy_return_mean_{window}"] = (
            spy_frame["spy_log_return_1d"]
            .rolling(window, min_periods=window)
            .mean()
        )
        spy_frame[f"spy_volatility_{window}"] = (
            spy_frame["spy_log_return_1d"]
            .rolling(window, min_periods=window)
            .std(ddof=0)
        )
    spy_frame[f"spy_trend_{spy_trend_window}"] = np.log(
        spy_frame["close"] / spy_frame["close"].shift(spy_trend_window)
    )
    spy_frame["spy_observed"] = np.int8(1)

    target_feature_columns = [
        column
        for column in target_frame.columns
        if column.startswith("asset_")
    ]
    spy_feature_columns = [
        column for column in spy_frame.columns if column.startswith("spy_")
    ]

    target_features = target_frame.loc[
        :, ["asset", "date", *target_feature_columns]
    ].rename(columns={"date": "origin_date"})
    spy_features = spy_frame.loc[:, ["date", *spy_feature_columns]].rename(
        columns={"date": "origin_date"}
    )

    result = target_features.merge(
        spy_features,
        on="origin_date",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    missing_mask = result["spy_observed"].isna()
    if missing_mask.any() and missing_spy == "raise":
        missing_dates = (
            result.loc[missing_mask, "origin_date"]
            .drop_duplicates()
            .sort_values()
            .dt.strftime("%Y-%m-%d")
            .head(10)
            .tolist()
        )
        raise DataValidationError(
            f"Target-stock rows have no same-date {spy_asset} observation: {missing_dates}."
        )
    result["spy_observed"] = result["spy_observed"].fillna(0).astype("int8")

    return result.sort_values(
        ["asset", "origin_date"], kind="mergesort", ignore_index=True
    )

