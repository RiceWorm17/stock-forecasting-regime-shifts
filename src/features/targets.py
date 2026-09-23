"""Independent next-observed-session target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import DataConfig
from src.data.validation import DataValidationError, validate_canonical_data


TARGET_COLUMNS: tuple[str, ...] = (
    "asset",
    "origin_date",
    "target_date",
    "target_log_return",
    "target_direction",
)


def generate_targets(
    canonical_data: pd.DataFrame,
    *,
    config: DataConfig | None = None,
) -> pd.DataFrame:
    """Create one-step targets within each asset using the next observed row.

    Input must already be explicitly normalized and sorted. The function returns
    target fields only; it does not create or join predictive features.
    """

    validate_canonical_data(
        canonical_data,
        config=config,
        require_sorted=True,
        require_all_assets=False,
    )

    working = canonical_data.loc[:, ["asset", "date", "close"]].copy()
    grouped = working.groupby("asset", sort=False, observed=True)
    working["next_date"] = grouped["date"].shift(-1)
    working["next_close"] = grouped["close"].shift(-1)
    supervised = working.loc[
        working["next_date"].notna() & working["next_close"].notna()
    ].copy()

    if supervised.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS).astype(
            {
                "asset": "string",
                "origin_date": "datetime64[ns]",
                "target_date": "datetime64[ns]",
                "target_log_return": "float64",
                "target_direction": "int8",
            }
        )

    if (supervised["next_date"] <= supervised["date"]).any():
        raise DataValidationError(
            "Every target_date must be strictly later than its origin_date within an asset."
        )
    if (supervised["close"] <= 0).any() or (supervised["next_close"] <= 0).any():
        raise DataValidationError("Positive close values are required for log-return targets.")

    target_log_return = np.log(
        supervised["next_close"].to_numpy(dtype="float64")
        / supervised["close"].to_numpy(dtype="float64")
    )
    result = pd.DataFrame(
        {
            "asset": supervised["asset"].astype("string").to_numpy(),
            "origin_date": pd.to_datetime(supervised["date"]).to_numpy(),
            "target_date": pd.to_datetime(supervised["next_date"]).to_numpy(),
            "target_log_return": target_log_return,
            "target_direction": (target_log_return > 0).astype("int8"),
        }
    )
    result = result.loc[:, list(TARGET_COLUMNS)].sort_values(
        ["asset", "origin_date", "target_date"], kind="mergesort", ignore_index=True
    )

    duplicate_identity = result.duplicated(
        ["asset", "origin_date", "target_date"], keep=False
    )
    if duplicate_identity.any():
        raise DataValidationError("Duplicate supervised-sample identities were generated.")
    return result

