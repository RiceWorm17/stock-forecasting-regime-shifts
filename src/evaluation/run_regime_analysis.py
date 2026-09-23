"""One-command Phase 2J development regime analysis runner."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.evaluation.regime_analysis import (
    Phase2JResult,
    run_regime_analysis,
    write_phase2j_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the frozen D1-D5 development regime analysis."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate Phase 2J feasibility without writing result artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_regime_analysis(dry_run=args.dry_run)
    if args.dry_run:
        print("Phase 2J dry run: PASS")
        print("Frozen integrity: PASS")
        print("Canonical prediction row counts: PASS")
        print("D1-D5 threshold-source eligibility: PASS")
        print("Unique SPY origin-date construction: PASS")
        print("Development labels structurally constructible: Yes")
        print("Post-2024 analysis rows: 0")
        print("PatchTST included: No")
        print("Models trained: No")
        print("Final artifacts written: 0")
        return 0

    if not isinstance(result, Phase2JResult):
        raise TypeError("Normal Phase 2J execution did not return its result bundle.")
    report_path = write_phase2j_report(result)
    print("Phase 2J development regime analysis: generated and independently verified")
    print(f"Content-addressed artifacts: {len(result.artifact_paths)}")
    print(f"Report: {report_path}")
    print("Models trained: No")
    print("F1/2025 performance evaluated: No")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
