from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.data.download import (
    download_locked_data,
    normalize_yfinance_frame,
    provider_end_exclusive,
    write_immutable_snapshot,
)
from src.data.schema import (
    CANONICAL_COLUMNS,
    TARGET_ASSETS,
    asset_role,
    load_data_config,
)
from src.data.validation import (
    DataValidationError,
    normalize_canonical_data,
    validate_canonical_data,
)


def test_locked_data_configuration_loads_deterministically():
    first = load_data_config()
    second = load_data_config()

    assert first == second
    assert first.assets == ("AAPL", "MSFT", "GOOGL", "NVDA")
    assert first.market_reference == ("SPY",)
    assert first.all_assets == ("AAPL", "MSFT", "GOOGL", "NVDA", "SPY")
    assert first.start_date.isoformat() == "2015-01-01"
    assert first.end_date.isoformat() == "2025-12-31"
    assert provider_end_exclusive(first) == "2026-01-01"
    assert first.auto_adjust is True


def test_target_and_market_reference_roles_are_distinct():
    assert all(asset_role(asset) == "target" for asset in TARGET_ASSETS)
    assert asset_role("SPY") == "market_reference"
    with pytest.raises(ValueError, match="outside the locked universe"):
        asset_role("TSLA")


def test_missing_required_column_fails(make_ohlcv):
    frame = make_ohlcv("AAPL", ["2020-01-02"], [100.0]).drop(columns="volume")
    with pytest.raises(DataValidationError, match="Missing required"):
        normalize_canonical_data(frame)


def test_duplicate_asset_date_fails(make_ohlcv):
    frame = make_ohlcv("AAPL", ["2020-01-02", "2020-01-02"], [100.0, 101.0])
    with pytest.raises(DataValidationError, match="Duplicate"):
        normalize_canonical_data(frame)


def test_unsorted_input_is_rejected_but_explicit_normalization_sorts(make_ohlcv):
    frame = make_ohlcv(
        "AAPL", ["2020-01-06", "2020-01-02", "2020-01-03"], [102.0, 100.0, 101.0]
    )
    with pytest.raises(DataValidationError, match="sorted"):
        validate_canonical_data(frame)

    normalized = normalize_canonical_data(frame)
    assert normalized["date"].tolist() == list(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("high", 99.0),
        ("low", 101.0),
    ],
)
def test_invalid_ohlc_relationships_fail(make_ohlcv, column, value):
    frame = normalize_canonical_data(make_ohlcv("AAPL", ["2020-01-02"], [100.0]))
    frame.loc[0, column] = value
    with pytest.raises(DataValidationError, match="OHLC relationship"):
        validate_canonical_data(frame)


def test_negative_volume_fails(make_ohlcv):
    frame = normalize_canonical_data(make_ohlcv("AAPL", ["2020-01-02"], [100.0]))
    frame.loc[0, "volume"] = -1.0
    with pytest.raises(DataValidationError, match="non-negative"):
        validate_canonical_data(frame)


def test_nonfinite_ohlc_fails(make_ohlcv):
    frame = make_ohlcv("AAPL", ["2020-01-02"], [100.0])
    frame.loc[0, "close"] = np.inf
    with pytest.raises(DataValidationError, match="non-finite"):
        normalize_canonical_data(frame)


def test_date_after_contract_fails(make_ohlcv):
    frame = make_ohlcv("AAPL", ["2026-01-02"], [100.0])
    with pytest.raises(DataValidationError, match="outside the locked range"):
        normalize_canonical_data(frame)


def test_unknown_asset_fails(make_ohlcv):
    frame = make_ohlcv("TSLA", ["2020-01-02"], [100.0])
    with pytest.raises(DataValidationError, match="outside the locked universe"):
        normalize_canonical_data(frame)


def test_yfinance_multiindex_normalization_uses_explicit_ticker_identity():
    columns = pd.MultiIndex.from_tuples(
        [(field, "AAPL") for field in ("Open", "High", "Low", "Close", "Volume")]
    )
    provider = pd.DataFrame(
        [[99.0, 102.0, 98.0, 101.0, 1_000.0]],
        index=pd.to_datetime(["2020-01-02"]),
        columns=columns,
    )

    result = normalize_yfinance_frame(provider, "AAPL")

    assert tuple(result.columns) == CANONICAL_COLUMNS
    assert result.loc[0, "asset"] == "AAPL"
    assert result.loc[0, "close"] == pytest.approx(101.0)


def test_download_function_requests_only_locked_assets_without_network():
    calls: list[tuple[str, dict]] = []

    def fake_download(asset: str, **kwargs):
        calls.append((asset, kwargs))
        return pd.DataFrame(
            {
                "Open": [99.0],
                "High": [102.0],
                "Low": [98.0],
                "Close": [101.0],
                "Volume": [1_000.0],
            },
            index=pd.to_datetime(["2020-01-02"]),
        )

    result = download_locked_data(download_function=fake_download)

    assert [asset for asset, _ in calls] == ["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"]
    assert all(call[1]["start"] == "2015-01-01" for call in calls)
    assert all(call[1]["end"] == "2026-01-01" for call in calls)
    assert all(call[1]["auto_adjust"] is True for call in calls)
    assert set(result["asset"]) == {"AAPL", "MSFT", "GOOGL", "NVDA", "SPY"}


def test_immutable_snapshot_has_hash_manifest_and_refuses_overwrite(
    tmp_path, make_ohlcv
):
    frames = [
        make_ohlcv(asset, ["2020-01-02"], [100.0 + index])
        for index, asset in enumerate(("AAPL", "MSFT", "GOOGL", "NVDA", "SPY"))
    ]
    frame = normalize_canonical_data(
        pd.concat(frames, ignore_index=True), require_all_assets=True
    )
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    snapshot = write_immutable_snapshot(
        frame,
        output_directory=tmp_path,
        retrieved_at=timestamp,
        provider_version="test-version",
    )

    assert snapshot.data.exists()
    assert snapshot.manifest.exists()
    assert hashlib.sha256(snapshot.data.read_bytes()).hexdigest() == snapshot.sha256
    manifest = json.loads(snapshot.manifest.read_text(encoding="utf-8"))
    assert manifest["file_sha256"] == snapshot.sha256
    assert manifest["provider"] == "yfinance"
    assert manifest["provider_library_version"] == "test-version"
    assert manifest["requested_range"]["provider_end_exclusive"] == "2026-01-01"
    assert manifest["row_count_by_asset"] == {
        "AAPL": 1,
        "GOOGL": 1,
        "MSFT": 1,
        "NVDA": 1,
        "SPY": 1,
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_immutable_snapshot(
            frame,
            output_directory=tmp_path,
            retrieved_at=timestamp,
            provider_version="test-version",
        )

