"""One-command, development-only execution of the locked Phase 2C baselines.

Run from the repository root with:

    python -m src.evaluation.run_baselines

The command consumes existing frozen raw and processed artifacts, verifies their
hash linkage, and evaluates only the guarded D1--D5 development test folds.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.integrity import file_sha256, verify_snapshot
from src.evaluation.baseline_evaluation import (
    BaselineArtifacts,
    run_and_save_development_baselines,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIRECTORY = REPOSITORY_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "baselines"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest {path} must contain a JSON object.")
    return payload


def _discover_one(directory: Path, pattern: str, label: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {label} in {directory}; found {len(candidates)}. "
            "Pass its manifest path explicitly."
        )
    return candidates[0]


def run_from_manifests(
    *,
    raw_manifest_path: str | Path,
    processed_manifest_path: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> BaselineArtifacts:
    """Verify frozen inputs and execute exactly one D1--D5 baseline run."""

    raw_manifest = Path(raw_manifest_path)
    raw_metadata = _load_json_object(raw_manifest)
    raw_filename = raw_metadata.get("canonical_file")
    if not isinstance(raw_filename, str) or not raw_filename:
        raise ValueError("Raw manifest has no canonical_file.")
    raw_data = raw_manifest.parent / raw_filename
    verified_raw = verify_snapshot(raw_data, raw_manifest)

    processed_manifest = Path(processed_manifest_path)
    processed_metadata = _load_json_object(processed_manifest)
    processed_filename = processed_metadata.get("processed_file")
    processed_digest = processed_metadata.get("file_sha256")
    if not isinstance(processed_filename, str) or not processed_filename:
        raise ValueError("Processed manifest has no processed_file.")
    if not isinstance(processed_digest, str) or len(processed_digest) != 64:
        raise ValueError("Processed manifest has no valid file_sha256.")
    processed_data = processed_manifest.parent / processed_filename
    actual_processed_digest = file_sha256(processed_data)
    if actual_processed_digest != processed_digest.lower():
        raise ValueError(
            "Processed data hash mismatch: "
            f"manifest={processed_digest}, actual={actual_processed_digest}."
        )

    source = processed_metadata.get("source")
    if not isinstance(source, dict):
        raise ValueError("Processed manifest source must be an object.")
    if source.get("raw_file_sha256") != verified_raw.sha256:
        raise ValueError("Processed manifest does not cite the verified raw SHA-256.")
    if source.get("raw_file") != raw_data.name:
        raise ValueError("Processed manifest does not cite the verified raw filename.")

    samples = pd.read_csv(processed_data)
    effective_run_id = run_id or f"phase2c_dev_{processed_digest[:12]}"
    effective_created_at = created_at or datetime.now(UTC)
    return run_and_save_development_baselines(
        samples,
        output_root=output_root,
        run_id=effective_run_id,
        source_snapshot_sha256=verified_raw.sha256,
        processed_dataset_sha256=processed_digest,
        created_at=effective_created_at,
        provenance={
            "raw_snapshot_file": raw_data.name,
            "raw_manifest_file": raw_manifest.name,
            "processed_dataset_file": processed_data.name,
            "processed_manifest_file": processed_manifest.name,
        },
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify frozen inputs and run only the D1-D5 zero-return and "
            "direction-persistence baselines."
        )
    )
    parser.add_argument("--raw-manifest", type=Path, default=None)
    parser.add_argument("--processed-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    raw_manifest = args.raw_manifest or _discover_one(
        DEFAULT_RAW_DIRECTORY, "*.manifest.json", "raw snapshot manifest"
    )
    processed_manifest = args.processed_manifest or _discover_one(
        DEFAULT_PROCESSED_DIRECTORY,
        "*.manifest.json",
        "processed dataset manifest",
    )
    artifacts = run_from_manifests(
        raw_manifest_path=raw_manifest,
        processed_manifest_path=processed_manifest,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    print(f"Created/verified predictions: {artifacts.prediction_path}")
    print(f"Created/verified per-asset metrics: {artifacts.per_asset_metrics_path}")
    print(f"Created/verified macro metrics: {artifacts.macro_metrics_path}")
    print(f"Created run manifest: {artifacts.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
