# Public Reproducibility Boundary

## Scope

This release candidate preserves a deliberately limited, inspectable subset of the completed research archive. It is designed to support review of the methodology, aggregate D1–D5 findings, model-selection records, verification manifests, and four published figures. It is not a bitwise end-to-end reproduction package.

## What can be reproduced from this package

- Inspect every aggregate result used by the README's D1–D5 scientific claims.
- Trace each primary claim through `docs/PUBLICATION_CLAIMS_LEDGER.md` to a content-addressed result under `results/`.
- Regenerate all four SVG figures from their four included aggregate CSV sources with:

```bash
python docs/figures/generate_figures.py
```

- Run the focused synthetic checks documented in the README. These checks exercise data schema, chronology, leakage guards, and walk-forward splitting without accessing the historical market snapshot or fitting a learned model.
- Inspect the implementation and frozen development configurations.

## What cannot be reproduced from this package

- The exact raw and processed market snapshots are not redistributed. Their content manifests are included, but a new provider download is not guaranteed to be byte-identical.
- Row-level development predictions and date-level regime labels are not redistributed pending a rights review. Consequently, a public-clone user cannot independently recalculate every historical aggregate metric from prediction rows.
- Trained LightGBM models, LSTM checkpoints, fitted scalers, candidate prediction trees, and machine-specific environment records are not included.
- The historical learned-model runs enforce frozen hashes, interpreter/environment constraints, and empty output locations. Their entry points are retained for inspection, not presented as portable quickstarts.
- There is no lockfile, container image, or vendored dependency set.
- F1/2025 artifacts are not included and must not be regenerated. The one-time run remains `INVALID_STOP_NO_RERUN`.

## Data acquisition boundary

Users must obtain market inputs independently and comply with the provider's terms. The project does not grant redistribution rights to third-party market data. The included manifests identify the historical provider, date ranges, schemas, row counts, and hashes; they do not make an independently downloaded snapshot an exact substitute.

## Evidence interpretation

The included aggregate evidence is sufficient to inspect the numbers and comparisons quoted in the README and to regenerate the four figures. Verification labels in the claims ledger describe checks completed against the preserved private archive. They should not be interpreted as a claim that the public candidate contains every input needed to repeat those checks.

