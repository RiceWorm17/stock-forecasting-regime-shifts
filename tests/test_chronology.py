from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import DataValidationError, normalize_canonical_data
from src.features.targets import generate_targets


def test_shuffling_before_explicit_normalization_does_not_change_targets(
    synthetic_market_data,
):
    expected = generate_targets(synthetic_market_data)
    shuffled = synthetic_market_data.sample(frac=1.0, random_state=42).reset_index(drop=True)

    normalized = normalize_canonical_data(shuffled, require_all_assets=True)
    actual = generate_targets(normalized)

    pd.testing.assert_frame_equal(actual, expected)


def test_target_generator_rejects_unsorted_input(make_ohlcv):
    unsorted = make_ohlcv(
        "AAPL", ["2020-01-06", "2020-01-02", "2020-01-03"], [102.0, 100.0, 101.0]
    )

    with pytest.raises(DataValidationError, match="sorted"):
        generate_targets(unsorted)


def test_normalized_rows_are_monotonic_within_every_asset(synthetic_market_data):
    for _, asset_frame in synthetic_market_data.groupby("asset", sort=False):
        assert asset_frame["date"].is_monotonic_increasing


def test_supervised_dates_are_strictly_chronological(synthetic_market_data):
    targets = generate_targets(synthetic_market_data)

    assert (targets["target_date"] > targets["origin_date"]).all()
    for _, asset_targets in targets.groupby("asset", sort=False):
        assert asset_targets["origin_date"].is_monotonic_increasing
        assert asset_targets["target_date"].is_monotonic_increasing

