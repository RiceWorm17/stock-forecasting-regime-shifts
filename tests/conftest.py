from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
import pytest

from src.data.validation import normalize_canonical_data


@pytest.fixture
def make_ohlcv():
    def _make(
        asset: str,
        dates: Sequence[str] | pd.DatetimeIndex,
        closes: Iterable[float],
        *,
        volumes: Iterable[float] | None = None,
    ) -> pd.DataFrame:
        close = np.asarray(list(closes), dtype="float64")
        open_values = close * 0.997
        volume_values = (
            np.asarray(list(volumes), dtype="float64")
            if volumes is not None
            else np.arange(len(close), dtype="float64") + 1_000.0
        )
        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "asset": asset,
                "open": open_values,
                "high": np.maximum(open_values, close) * 1.01,
                "low": np.minimum(open_values, close) * 0.99,
                "close": close,
                "volume": volume_values,
            }
        )

    return _make


@pytest.fixture
def synthetic_market_data(make_ohlcv) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=90)
    frames = []
    for index, asset in enumerate(("AAPL", "MSFT", "GOOGL", "NVDA", "SPY")):
        base = 100.0 + index * 25.0
        closes = base * np.exp(np.linspace(0.0, 0.15 + index * 0.01, len(dates)))
        frames.append(make_ohlcv(asset, dates, closes))
    return normalize_canonical_data(pd.concat(frames, ignore_index=True), require_all_assets=True)

