"""Locked non-learned baselines for next-session return and direction."""

from __future__ import annotations

import numpy as np
import pandas as pd


REGRESSION_BASELINE_ID = "zero_return"
DIRECTION_BASELINE_ID = "direction_persistence"
COMBINED_BASELINE_ID = f"{REGRESSION_BASELINE_ID}__{DIRECTION_BASELINE_ID}"


def zero_return_prediction(index: pd.Index) -> pd.Series:
    """Return the preregistered zero next-day log-return prediction."""

    return pd.Series(0.0, index=index, dtype="float64")


def direction_persistence_prediction(current_asset_log_return: pd.Series) -> pd.Series:
    """Predict the next direction as the sign class of the current asset return.

    Direction uses the project's binary target convention: strictly positive is
    class 1, while zero and negative returns are class 0.
    """

    numeric = pd.to_numeric(current_asset_log_return, errors="raise")
    values = numeric.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("Current asset log returns must all be finite.")
    return pd.Series((values > 0).astype("int8"), index=current_asset_log_return.index)

