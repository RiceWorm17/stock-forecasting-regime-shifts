"""One-command Phase 2G combined development benchmark audit."""

from __future__ import annotations

from src.evaluation.combined_audit import run_combined_audit


def main() -> int:
    result = run_combined_audit()
    print("Phase 2G combined audit artifacts:")
    for name, path in result.artifact_paths.items():
        print(f"  {name}: {path} ({result.artifact_hashes[name]})")
    print(f"Canonical development keys: {result.key_audit['unique_keys']}")
    print(f"Combined fold/asset groups: {result.summary['asset_fold_groups']}/20")
    print(f"PatchTST decision: {result.gate_decision['decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
