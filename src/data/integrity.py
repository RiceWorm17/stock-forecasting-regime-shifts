"""Verification for immutable, content-addressed raw-data snapshots.

The downloader writes a canonical CSV and a sidecar JSON manifest.  This module
recomputes every manifest statistic from the stored bytes and data instead of
trusting the sidecar.  It performs no repair and never writes to the snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.data.schema import CANONICAL_COLUMNS, DataConfig, load_data_config
from src.data.validation import DataValidationError, validate_canonical_data


class SnapshotIntegrityError(ValueError):
    """Raised when snapshot bytes, data, or provenance disagree."""


@dataclass(frozen=True)
class SnapshotVerification:
    """Verified facts recomputed from a frozen snapshot."""

    data_path: Path
    manifest_path: Path
    sha256: str
    total_rows: int
    row_count_by_asset: dict[str, int]
    returned_range_by_asset: dict[str, dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable verification record."""

        return {
            "data_path": str(self.data_path),
            "manifest_path": str(self.manifest_path),
            "sha256": self.sha256,
            "total_rows": self.total_rows,
            "row_count_by_asset": dict(self.row_count_by_asset),
            "returned_range_by_asset": {
                asset: dict(bounds)
                for asset, bounds in self.returned_range_by_asset.items()
            },
        }


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 from the exact bytes stored on disk."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SnapshotIntegrityError(f"Unable to read snapshot bytes {path}: {exc}") from exc
    return digest.hexdigest()


def load_snapshot_manifest(path: str | Path) -> dict[str, Any]:
    """Load a manifest and require a JSON object at its root."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SnapshotIntegrityError(
            f"Unable to read snapshot manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SnapshotIntegrityError("Snapshot manifest root must be a JSON object.")
    return payload


def _require_mapping(
    manifest: Mapping[str, Any], key: str
) -> Mapping[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise SnapshotIntegrityError(f"Manifest field {key!r} must be an object.")
    return value


def _expected_contract(config: DataConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": config.provider,
        "frequency": config.frequency,
        "timezone": config.timezone,
    }


def _verify_manifest_contract(
    manifest: Mapping[str, Any], data_path: Path, config: DataConfig
) -> None:
    for key, expected in _expected_contract(config).items():
        if manifest.get(key) != expected:
            raise SnapshotIntegrityError(
                f"Manifest {key!r} is {manifest.get(key)!r}; expected {expected!r}."
            )

    if manifest.get("canonical_file") != data_path.name:
        raise SnapshotIntegrityError(
            "Manifest canonical_file does not name the supplied snapshot: "
            f"{manifest.get('canonical_file')!r} != {data_path.name!r}."
        )

    requested = _require_mapping(manifest, "requested_range")
    expected_requested = {
        "start_inclusive": config.start_date.isoformat(),
        "end_inclusive": config.end_date.isoformat(),
        "provider_end_exclusive": (config.end_date + timedelta(days=1)).isoformat(),
    }
    if dict(requested) != expected_requested:
        raise SnapshotIntegrityError(
            "Manifest requested_range differs from the locked contract: "
            f"{dict(requested)!r} != {expected_requested!r}."
        )

    adjustment = _require_mapping(manifest, "adjustment_setting")
    if dict(adjustment) != {"auto_adjust": config.auto_adjust}:
        raise SnapshotIntegrityError(
            "Manifest adjustment_setting differs from the locked contract."
        )


def _read_canonical_snapshot(path: Path, config: DataConfig) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise SnapshotIntegrityError(f"Unable to parse snapshot {path}: {exc}") from exc

    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != CANONICAL_COLUMNS:
        raise SnapshotIntegrityError(
            "Snapshot columns must exactly match the canonical ordered schema: "
            f"{actual_columns!r} != {CANONICAL_COLUMNS!r}."
        )

    raw_assets = frame["asset"].astype("string")
    if not raw_assets.isin(config.all_assets).all():
        bad = sorted(set(raw_assets.dropna().astype(str)).difference(config.all_assets))
        raise SnapshotIntegrityError(
            "Snapshot asset values are not canonical locked tickers; "
            f"unexpected values: {bad!r}."
        )

    # The frozen writer emits one unambiguous ISO session date per row.
    date_text = frame["date"].astype("string")
    iso_mask = date_text.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    if not iso_mask.all():
        examples = date_text.loc[~iso_mask].head(5).tolist()
        raise SnapshotIntegrityError(
            f"Snapshot dates must be canonical YYYY-MM-DD strings; found {examples!r}."
        )

    try:
        validate_canonical_data(
            frame,
            config=config,
            require_sorted=True,
            require_all_assets=True,
        )
    except DataValidationError as exc:
        raise SnapshotIntegrityError(f"Snapshot data validation failed: {exc}") from exc
    return frame


def _recompute_statistics(
    frame: pd.DataFrame, config: DataConfig
) -> tuple[dict[str, int], dict[str, dict[str, str]]]:
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    working = frame.assign(date=dates)
    row_counts: dict[str, int] = {}
    returned_ranges: dict[str, dict[str, str]] = {}
    for asset in config.all_assets:
        asset_rows = working.loc[working["asset"] == asset]
        row_counts[asset] = int(len(asset_rows))
        returned_ranges[asset] = {
            "start": asset_rows["date"].min().date().isoformat(),
            "end": asset_rows["date"].max().date().isoformat(),
        }
    return row_counts, returned_ranges


def verify_snapshot(
    data_path: str | Path,
    manifest_path: str | Path,
    *,
    config: DataConfig | None = None,
) -> SnapshotVerification:
    """Verify bytes, canonical data, counts, ranges, and locked provenance.

    Both paths are explicit so callers cannot accidentally associate a data file
    with whichever manifest happens to sort first in a directory.
    """

    active_config = config or load_data_config()
    csv_path = Path(data_path)
    sidecar_path = Path(manifest_path)
    manifest = load_snapshot_manifest(sidecar_path)
    _verify_manifest_contract(manifest, csv_path, active_config)

    expected_digest = manifest.get("file_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise SnapshotIntegrityError(
            "Manifest file_sha256 must be a 64-character hexadecimal string."
        )
    try:
        int(expected_digest, 16)
    except ValueError as exc:
        raise SnapshotIntegrityError("Manifest file_sha256 is not hexadecimal.") from exc

    actual_digest = file_sha256(csv_path)
    if actual_digest != expected_digest.lower():
        raise SnapshotIntegrityError(
            "Snapshot SHA-256 mismatch: "
            f"manifest={expected_digest!r}, actual={actual_digest!r}."
        )

    frame = _read_canonical_snapshot(csv_path, active_config)
    row_counts, returned_ranges = _recompute_statistics(frame, active_config)

    manifest_counts = _require_mapping(manifest, "row_count_by_asset")
    if dict(manifest_counts) != row_counts:
        raise SnapshotIntegrityError(
            "Manifest row_count_by_asset does not match snapshot data: "
            f"{dict(manifest_counts)!r} != {row_counts!r}."
        )

    manifest_ranges = _require_mapping(manifest, "returned_range_by_asset")
    if dict(manifest_ranges) != returned_ranges:
        raise SnapshotIntegrityError(
            "Manifest returned_range_by_asset does not match snapshot data: "
            f"{dict(manifest_ranges)!r} != {returned_ranges!r}."
        )

    return SnapshotVerification(
        data_path=csv_path,
        manifest_path=sidecar_path,
        sha256=actual_digest,
        total_rows=int(len(frame)),
        row_count_by_asset=row_counts,
        returned_range_by_asset=returned_ranges,
    )
