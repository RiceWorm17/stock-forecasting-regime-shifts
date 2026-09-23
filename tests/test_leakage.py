from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data.validation import DataValidationError, normalize_canonical_data
from src.features.market_features import build_market_features


def test_future_perturbation_cannot_change_past_features(synthetic_market_data):
    cutoff = pd.Timestamp(synthetic_market_data["date"].sort_values().unique()[45])
    before = build_market_features(
        synthetic_market_data, rolling_windows=(3, 5), spy_trend_window=5
    )

    perturbed = synthetic_market_data.copy()
    future = perturbed["date"] > cutoff
    for column in ("open", "high", "low", "close"):
        perturbed.loc[future, column] *= 1.75
    perturbed.loc[future, "volume"] *= 2.0
    perturbed = normalize_canonical_data(perturbed, require_all_assets=True)
    after = build_market_features(
        perturbed, rolling_windows=(3, 5), spy_trend_window=5
    )

    before_past = before.loc[before["origin_date"] <= cutoff].reset_index(drop=True)
    after_past = after.loc[after["origin_date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(before_past, after_past)


def test_rolling_features_use_only_current_and_prior_observations(make_ohlcv):
    dates = pd.bdate_range("2020-01-02", periods=5)
    aapl = make_ohlcv("AAPL", dates, [100.0, 110.0, 121.0, 133.1, 146.41])
    spy = make_ohlcv("SPY", dates, [200.0, 202.0, 204.02, 206.0602, 208.120802])
    canonical = normalize_canonical_data(pd.concat([aapl, spy], ignore_index=True))

    features = build_market_features(
        canonical, rolling_windows=(2,), spy_trend_window=2
    )
    third = features.loc[
        (features["asset"] == "AAPL")
        & (features["origin_date"] == pd.Timestamp(dates[2]))
    ].iloc[0]

    expected_return = math.log(1.1)
    assert third["asset_log_return_1d"] == pytest.approx(expected_return)
    assert third["asset_log_return_lag_1"] == pytest.approx(expected_return)
    assert third["asset_return_mean_2"] == pytest.approx(expected_return)
    assert third["asset_return_std_2"] == pytest.approx(0.0, abs=1e-12)


def test_spy_features_join_by_date_not_row_position(make_ohlcv):
    dates = pd.bdate_range("2020-01-02", periods=4)
    aapl = make_ohlcv("AAPL", dates, [10.0, 11.0, 12.0, 13.0])
    msft = make_ohlcv("MSFT", dates, [30.0, 29.0, 28.0, 27.0])
    spy = make_ohlcv("SPY", dates, [100.0, 200.0, 100.0, 50.0])
    mixed = pd.concat([msft.iloc[::-1], spy, aapl.iloc[::-1]], ignore_index=True)
    canonical = normalize_canonical_data(mixed)

    features = build_market_features(
        canonical, rolling_windows=(2,), spy_trend_window=2
    )
    on_second_date = features.loc[features["origin_date"] == dates[1]]

    assert set(on_second_date["asset"]) == {"AAPL", "MSFT"}
    assert np.allclose(
        on_second_date["spy_log_return_1d"].to_numpy(),
        math.log(2.0),
        rtol=1e-12,
        atol=1e-12,
    )
    assert (on_second_date["spy_observed"] == 1).all()


def test_missing_spy_date_is_explicit_and_never_future_filled(make_ohlcv):
    target_dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    spy_dates = pd.to_datetime(["2020-01-02", "2020-01-06"])
    aapl = make_ohlcv("AAPL", target_dates, [100.0, 101.0, 102.0])
    spy = make_ohlcv("SPY", spy_dates, [200.0, 220.0])
    canonical = normalize_canonical_data(pd.concat([aapl, spy], ignore_index=True))

    with pytest.raises(DataValidationError, match="no same-date SPY"):
        build_market_features(
            canonical, rolling_windows=(2,), spy_trend_window=2, missing_spy="raise"
        )

    kept = build_market_features(
        canonical, rolling_windows=(2,), spy_trend_window=2, missing_spy="keep"
    )
    missing_row = kept.loc[kept["origin_date"] == pd.Timestamp("2020-01-03")].iloc[0]
    assert missing_row["spy_observed"] == 0
    assert np.isnan(missing_row["spy_log_return_1d"])


def test_feature_module_does_not_generate_targets(synthetic_market_data):
    features = build_market_features(
        synthetic_market_data, rolling_windows=(3,), spy_trend_window=3
    )
    assert not any(column.startswith("target_") for column in features.columns)
