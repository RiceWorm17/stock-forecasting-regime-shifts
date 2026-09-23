# Public File Manifest

## Decision and scope

This manifest defines the allowlist for the GitHub release candidate finalized on 2026-09-23. Classification is conservative:

- **PUBLIC:** copied into the separate release-candidate directory.
- **EXCLUDE:** retained only in the private research archive and not needed in the public candidate.
- **REVIEW_REQUIRED:** must not be published unless the owner resolves the stated rights, privacy, or provenance question.

Only paths listed under **PUBLIC allowlist** are approved for staging. Everything else in the private archive is excluded unless explicitly listed as review-required. No classification is an independent legal determination.

The candidate contains 98 files after creation of `PUBLIC_FILES.sha256`. That checksum inventory covers every staged file except itself. All 28 content-addressed result files below were independently hashed, and each complete filename hash matched its file bytes.

## PUBLIC allowlist

### Project metadata and selected reports

```text
.gitattributes
.gitignore
PUBLIC_FILES.sha256
README.md
pyproject.toml
PHASE_2G_REPORT.md
PHASE_2H_C_REPORT.md
PHASE_2J_REPORT.md
PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md
```

The staged `README.md` contains publication-only availability edits. The four phase reports are unchanged copies from the private archive.

### Configurations

```text
configs/data.yaml
configs/model_search.yaml
configs/patchtst_search.yaml
configs/regime_analysis.yaml
configs/walk_forward.yaml
```

### Data provenance without market rows

```text
data/processed/supervised_v1_9352da6b434011f9844ef0acc1813ea33dce39152712752bba21a3005aabf7be.manifest.json
data/raw/yfinance_daily_2015-01-01_2025-12-31_20260913T152313Z_55f74f058493.manifest.json
```

These manifests identify schemas, ranges, counts, and data hashes. They do not contain the underlying market observations.

### Documentation and figures

```text
docs/ATTRIBUTION_AND_PROVENANCE.md
docs/DATA_CONTRACT.md
docs/EXPERIMENT_SPEC.md
docs/EXPERIMENT_SPEC_CHANGELOG.md
docs/F1_INVALID_DISCLOSURE.md
docs/LICENSE_STATUS.md
docs/MODEL_PREREGISTRATION.md
docs/PATCHTST_PREREGISTRATION.md
docs/PUBLIC_FILE_MANIFEST.md
docs/PUBLIC_REPRODUCIBILITY.md
docs/PUBLICATION_CLAIMS_LEDGER.md
docs/REGIME_PREREGISTRATION.md
docs/figures/README.md
docs/figures/asset_fold_mae_skill.svg
docs/figures/generate_figures.py
docs/figures/macro_mae_skill_by_fold.svg
docs/figures/regime_mae_skill_degradation.svg
docs/figures/regime_reportability.svg
```

`docs/EXPERIMENT_SPEC.md` and `docs/PUBLICATION_CLAIMS_LEDGER.md` contain publication-only availability notes in staging. The four SVGs and their generator are unchanged copies from the private archive.

### Source

```text
src/__init__.py
src/data/__init__.py
src/data/calendar_audit.py
src/data/download.py
src/data/integrity.py
src/data/process.py
src/data/schema.py
src/data/validation.py
src/evaluation/__init__.py
src/evaluation/baseline_evaluation.py
src/evaluation/combined_audit.py
src/evaluation/lightgbm_evaluation.py
src/evaluation/lstm_evaluation.py
src/evaluation/patchtst_pilot.py
src/evaluation/regime_analysis.py
src/evaluation/run_baselines.py
src/evaluation/run_combined_audit.py
src/evaluation/run_lightgbm.py
src/evaluation/run_lstm.py
src/evaluation/run_patchtst_pilot.py
src/evaluation/run_regime_analysis.py
src/evaluation/walk_forward.py
src/features/__init__.py
src/features/market_features.py
src/features/targets.py
src/models/__init__.py
src/models/baselines.py
src/models/lightgbm_model.py
src/models/lstm_model.py
src/models/patchtst_model.py
src/utils/__init__.py
```

### Focused synthetic tests

```text
tests/conftest.py
tests/test_chronology.py
tests/test_data_schema.py
tests/test_leakage.py
tests/test_walk_forward.py
```

### Public-safe aggregate and provenance results

```text
results/baselines/run_manifest_9927474b9a3ef014ad4e016c48a7f8cda461a0ccb3e0daaab6f124135ddc32d0.json
results/combined/baseline_win_loss_counts_0e5210dfbf806fec4a065e138e115524c317242ed3a7c6e1b2e8068b3e858684.csv
results/combined/combined_verification_manifest_393be15393847bdcc48d9482a0d72f32437f1024e85523088dbc3194c4eabb02.json
results/combined/cross_fold_summary_5d6eeac0aad1b87b735714f0238f711b97fa6abce5b77608e42b0c905d3daad8.csv
results/combined/development_asset_fold_comparison_61ad31f5e745338ccf679be9614188bab869964b350ab4f32a26067f85a2f3a1.csv
results/combined/development_fold_ranking_e13f0bd43d0b2f3ae80e41f8d0278e808a1cf04509b1eda8c1986e13d6826de9.csv
results/combined/development_macro_fold_comparison_042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6.csv
results/combined/lstm_seed_stability_f965911f3b488f43e12e34996cef785e40de4d2d54fa52485f3e2f4064197d95.csv
results/combined/model_complexity_summary_e5d4ce6f88a686663da21b188f86bae92614075580ac7722486ab0844f8c05d6.csv
results/combined/patchtst_gate_decision_b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7.json
results/lightgbm/selections/lightgbm_fold_candidates_682a285d3004c5de8545a38aa9bfdd8972b3155509227e7f92f2164c58662b5f.csv
results/lightgbm/selections/lightgbm_winning_best_iterations_67e9104472c03c08971cd795e898b3413da06a56fda491895e9834cd17ae82c9.csv
results/lstm/selections/lstm_fold_candidates_e6d2c61026d6fdb2d1756c7d6ec520915279cb3dbb2d8342ec198ee0dd514187.csv
results/lstm/selections/lstm_winning_best_epochs_f4c38527e969d9242bf33b24220e2880ab087ea113c684783c7ea13ca85718a3.csv
results/regime/development/asset_fold_regime_degradation_278dc64486ce255773a578bf6c285d7831e4f846c156ffab2a08ca99d45495eb.csv
results/regime/development/baseline_regime_cell_metrics_e965db85359751403417205fa619584b6153d07f3854f342baa94e1bda12e0dc.csv
results/regime/development/cross_fold_regime_summary_046a0816d7940d5cb3b4b90817f9353024c3d3e0a60cce715efe98e4efded6ca.csv
results/regime/development/development_regime_label_counts_45881ac07a118218bc4128e5e9975cd5dcc0410967ec491447fd3f9a2570ee26.csv
results/regime/development/directional_error_disagreement_9330d2595da96b6f87f3c32585326873ace2f6aab738c58dbf9d3604fa6c35eb.csv
results/regime/development/lightgbm_regime_cell_metrics_615e9464de3327d3d297fb718907bb71cdd5a6cf4de178d8afc3f0c1099b4c79.csv
results/regime/development/lstm_per_seed_regime_cell_metrics_9305b93729681f279c8b17032ee5ef5dbee7c50f1390a9ddcb2409928945d0ac.csv
results/regime/development/lstm_seed_summary_regime_metrics_b2f81fde8a0a7f5ebc01068efd376096609ca5f5a671e3ef5a6e1c278b4265b8.csv
results/regime/development/macro_regime_degradation_55c49a10739c7fcdc430ad506edf8d2c172d9843da4de043e82080e97663c80a.csv
results/regime/development/macro_regime_metrics_d892a985633c11f68bb500347ec6036166ce65e58df4544508357eeb9bcbe2fd.csv
results/regime/development/phase2j_independent_verification_d19ad9a5f41e904e591803b5c6f73617f7eafe27dcd9cd798f599c8e1f5573e1.json
results/regime/development/phase2j_run_manifest_6e0aa516b2a5e21918cf5c6f97c641ac42ca54097865602867d27c6ff0647759.json
results/regime/development/regime_failure_pattern_summary_68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f.csv
results/regime/development/regime_threshold_audit_6f8ccb0a90c8bb9419061aa60cc6bcce781a26bcae5577f09cfa3a35920aa333.csv
```

These 28 files are aggregate tables or path-free provenance records. They total 503,736 bytes. The four figure sources are the development macro comparison, development asset/fold comparison, asset/fold regime degradation, and regime failure-pattern summary CSVs.

## EXCLUDE from the public candidate

| Private-archive path or group | Reason |
|---|---|
| `assignment.pdf` | Third-party course material; author metadata and redistribution permission are unresolved. |
| `original_submission.zip` | Contains a possible student identifier, raw/results material, hard-coded local paths, OS metadata, and no redistribution grant. |
| `desktop.ini`, `.pytest_cache/`, every `__pycache__/`, and every `*.pyc` | Local system/cache artifacts with no publication value. |
| `data/raw/*.csv` and `data/processed/*.csv` | Provider-sourced market rows and derived row-level values lack a documented redistribution grant. |
| Row-level prediction CSVs and date-level regime-label CSVs anywhere under `results/` | Derived market rows remain subject to the same conservative rights review; aggregate evidence is provided instead. |
| Trained LightGBM models, PyTorch checkpoints, fitted scalers, candidate prediction trees, and full historical output trees | Unnecessary for the published aggregate claims; ownership/redistribution and unsafe checkpoint-loading concerns remain. |
| `results/final_test/**` | Private F1 governance, prediction, metric, signing, and invalid-state archive; F1 is invalid and no public claim depends on these files. |
| Machine-specific environment/run records containing user-profile paths | Privacy/reproducibility risk; path-free public summaries are used instead. |
| `configs/final_test_execution.yaml`, `src/evaluation/final_test_evaluation.py`, and `src/evaluation/run_final_test.py` | One-time F1 implementation and machine-specific boundary are not part of the public rerunnable surface. |
| Tests outside the five-file allowlist above | They require withheld snapshots/artifacts, fit models, or exercise the private final-test/governance system. |
| `AUDIT_REPORT.md`, unselected `PHASE_2*.md` files, final-test governance documents, and `docs/GRADUATE_APPLICATION_PROJECT_SUMMARY.md` | Internal audit/governance history or application-draft material is unnecessary for the scientific public candidate; some files contain local paths or unresolved personal/provenance context. |

## Resolved release decisions

| Item | Resolution |
|---|---|
| Project provenance | The owner states that they developed the public project with AI assistance, accepts responsibility for the final contents, and authorizes publication of this conservative candidate. |
| Project license | Deliberate no-license publication. No `LICENSE` exists, the project is not described as open source, and public visibility is not presented as granting reuse rights. |
| Third-party dependencies and methods | Dependencies are referenced rather than vendored; upstream license/source links and LSTM, LightGBM, PatchTST, yfinance, and data-rights acknowledgments are recorded in `docs/ATTRIBUTION_AND_PROVENANCE.md`. |

## Rights-sensitive material that remains excluded

| Item | Condition for any future inclusion |
|---|---|
| Market data and row-level derivatives | Obtain an affirmative provider/exchange redistribution determination before adding raw, processed, prediction, or date-level label rows. |
| Course assignment and original submission | Confirm ownership, instructor/course permission, collaborators, and removal of personal metadata before publication. |
| Model weights/checkpoints | Confirm ownership and intended redistribution terms; they are unnecessary for this candidate. |

No included file currently depends on an unresolved redistribution permission for one of these excluded materials.

## Evidence boundary

All aggregate D1–D5 result paths cited for the README's primary findings are present. The public candidate does not contain row-level evidence needed to independently reconstruct those metrics, and it says so in the README, claims ledger, and reproducibility note. F1/2025 evidence remains private and invalid; only the unchanged forensic report and a public disclosure are included.

