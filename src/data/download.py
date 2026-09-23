"""Explicit, immutable acquisition of the locked daily yfinance dataset.

Importing this module never performs network I/O. Run it explicitly with:

    python -m src.data.download
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import CANONICAL_COLUMNS, DataConfig, load_data_config
from src.data.validation import (
    DataValidationError,
    normalize_canonical_data,
    validate_canonical_data,
)


DEFAULT_RAW_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass(frozen=True)
class SnapshotPaths:
    data: Path
    manifest: Path
    sha256: str


def provider_end_exclusive(config: DataConfig) -> str:
    """Return the day after the inclusive contract end for yfinance requests."""

    return (config.end_date + timedelta(days=1)).isoformat()


def _select_ticker_columns(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Select one ticker from either common yfinance column orientation."""

    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy()

    normalized_asset = asset.upper()
    matching_levels: list[int] = []
    for level in range(frame.columns.nlevels):
        values = {str(value).strip().upper() for value in frame.columns.get_level_values(level)}
        if normalized_asset in values:
            matching_levels.append(level)

    if len(matching_levels) != 1:
        raise DataValidationError(
            f"Could not identify ticker {asset!r} uniquely in provider columns."
        )

    selected = frame.xs(asset, axis=1, level=matching_levels[0], drop_level=True)
    if isinstance(selected, pd.Series):
        selected = selected.to_frame()
    if isinstance(selected.columns, pd.MultiIndex):
        selected.columns = selected.columns.get_level_values(0)
    return selected.copy()


def normalize_yfinance_frame(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Normalize one explicitly requested ticker into the canonical raw schema."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataValidationError(f"Provider returned no rows for {asset}.")

    selected = _select_ticker_columns(frame, asset)
    renamed = {
        column: str(column).strip().lower().replace(" ", "_")
        for column in selected.columns
    }
    selected = selected.rename(columns=renamed)

    required_provider_fields = {"open", "high", "low", "close", "volume"}
    missing = required_provider_fields.difference(selected.columns)
    if missing:
        raise DataValidationError(
            f"Provider response for {asset} is missing fields: {sorted(missing)}."
        )

    if "date" in selected.columns:
        dates = selected["date"]
    elif "datetime" in selected.columns:
        dates = selected["datetime"]
    else:
        dates = pd.Series(selected.index, index=selected.index)

    canonical = pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "asset": asset,
            "open": selected["open"].to_numpy(),
            "high": selected["high"].to_numpy(),
            "low": selected["low"].to_numpy(),
            "close": selected["close"].to_numpy(),
            "volume": selected["volume"].to_numpy(),
        }
    )
    return normalize_canonical_data(canonical, require_all_assets=False)


def _default_yfinance_download() -> Callable[..., pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for acquisition. Install the declared project dependencies."
        ) from exc
    return yf.download


def download_locked_data(
    config: DataConfig | None = None,
    *,
    download_function: Callable[..., pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Download every locked ticker explicitly without writing or mutating a snapshot."""

    active_config = config or load_data_config()
    provider_download = download_function or _default_yfinance_download()
    provider_end = provider_end_exclusive(active_config)
    normalized_frames: list[pd.DataFrame] = []

    for asset in active_config.all_assets:
        response = provider_download(
            asset,
            start=active_config.start_date.isoformat(),
            end=provider_end,
            interval="1d",
            auto_adjust=active_config.auto_adjust,
            actions=False,
            progress=False,
            threads=False,
        )
        normalized_frames.append(normalize_yfinance_frame(response, asset))

    combined = pd.concat(normalized_frames, ignore_index=True)
    combined = normalize_canonical_data(
        combined, config=active_config, require_all_assets=True
    )
    validate_canonical_data(
        combined,
        config=active_config,
        require_sorted=True,
        require_all_assets=True,
    )
    return combined


def _canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    serializable = frame.loc[:, list(CANONICAL_COLUMNS)].copy()
    serializable["date"] = pd.to_datetime(serializable["date"]).dt.strftime("%Y-%m-%d")
    text = serializable.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    return text.encode("utf-8")


def _provider_version() -> str | None:
    try:
        return importlib.metadata.version("yfinance")
    except importlib.metadata.PackageNotFoundError:
        return None


def write_immutable_snapshot(
    frame: pd.DataFrame,
    *,
    output_directory: str | Path = DEFAULT_RAW_DIRECTORY,
    config: DataConfig | None = None,
    retrieved_at: datetime | None = None,
    provider_version: str | None = None,
) -> SnapshotPaths:
    """Write a versioned canonical CSV and manifest using exclusive creation."""

    active_config = config or load_data_config()
    canonical = normalize_canonical_data(
        frame, config=active_config, require_all_assets=True
    )
    payload = _canonical_csv_bytes(canonical)
    digest = hashlib.sha256(payload).hexdigest()

    timestamp = retrieved_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware.")
    timestamp = timestamp.astimezone(UTC)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = (
        f"yfinance_daily_{active_config.start_date.isoformat()}_"
        f"{active_config.end_date.isoformat()}_{stamp}_{digest[:12]}"
    )
    data_path = output_path / f"{stem}.csv"
    manifest_path = output_path / f"{stem}.manifest.json"

    returned_ranges: dict[str, dict[str, Any]] = {}
    row_counts: dict[str, int] = {}
    for asset, asset_frame in canonical.groupby("asset", sort=True):
        returned_ranges[str(asset)] = {
            "start": pd.Timestamp(asset_frame["date"].min()).date().isoformat(),
            "end": pd.Timestamp(asset_frame["date"].max()).date().isoformat(),
        }
        row_counts[str(asset)] = int(len(asset_frame))

    manifest = {
        "schema_version": 1,
        "provider": active_config.provider,
        "provider_library_version": provider_version
        if provider_version is not None
        else _provider_version(),
        "retrieval_timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "requested_range": {
            "start_inclusive": active_config.start_date.isoformat(),
            "end_inclusive": active_config.end_date.isoformat(),
            "provider_end_exclusive": provider_end_exclusive(active_config),
        },
        "returned_range_by_asset": returned_ranges,
        "row_count_by_asset": row_counts,
        "canonical_file": data_path.name,
        "file_sha256": digest,
        "adjustment_setting": {"auto_adjust": active_config.auto_adjust},
        "frequency": active_config.frequency,
        "timezone": active_config.timezone,
    }

    try:
        with data_path.open("xb") as handle:
            handle.write(payload)
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Immutable snapshot already exists; refusing to overwrite {exc.filename}."
        ) from exc

    return SnapshotPaths(data=data_path, manifest=manifest_path, sha256=digest)


def acquire_snapshot(
    *,
    config_path: str | Path | None = None,
    output_directory: str | Path = DEFAULT_RAW_DIRECTORY,
) -> SnapshotPaths:
    """Explicit network acquisition entry point used only by the CLI."""

    config = load_data_config(config_path)
    retrieved_at = datetime.now(UTC)
    frame = download_locked_data(config)
    return write_immutable_snapshot(
        frame,
        output_directory=output_directory,
        config=config,
        retrieved_at=retrieved_at,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explicitly acquire the locked, immutable V2 market-data snapshot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the locked data YAML (defaults to configs/data.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIRECTORY,
        help="Directory for the versioned CSV and manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = acquire_snapshot(config_path=args.config, output_directory=args.output_dir)
    print(f"Created immutable snapshot: {paths.data}")
    print(f"Created manifest: {paths.manifest}")
    print(f"SHA-256: {paths.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
