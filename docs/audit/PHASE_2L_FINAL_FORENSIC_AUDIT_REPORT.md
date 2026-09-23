# Phase 2L Final Post-Failure Forensic Audit

Project: **When Stock Forecasting Models Fail: Stress-Testing Robustness Under Market Regime Shifts**  
Audit scope: **one bounded, read-only post-failure audit; no scientific re-execution**  
Audit date: **2026-09-18**

## 1. Executive summary

The original one-time F1/2025 execution remains **INVALID**. Its authorization was irreversibly consumed, target access occurred once, and a retry is prohibited. This audit did not execute F1, fit a model, regenerate a prediction, call a production evaluation function, alter a verifier, or modify an existing scientific file.

The execution generated and saved its sealed predictions, canonical raw and labeled predictions, overall metrics, regime labels, regime metrics, degradation tables, low-sample audit, and development-versus-F1 comparison. It then failed during the required independent saved-file verification at the exact sealed-versus-canonical prediction equality check. No final independent-verification artifact, final run manifest, or normal completion report was produced.

The failure mechanism is **confirmed**. The sealed learned-model values were written with `float_format="%.17g"`, reloaded by the default pandas CSV parser, copied into the canonical frames, and written again with the same format. On every row, the canonical decimal text equals the `.17g` serialization of the default-parser value loaded from the sealed file. When the canonical text is read with round-trip parsing, it exactly recovers that intermediate value. A second default-parser read changes some last-place bits, causing the frozen `numpy.array_equal` check to fail. This is a parse-and-reserialize defect, not a key, row, target, model-identity, or direction mismatch.

The independently measured discrepancy is numerically tiny but governance-controlling:

- baseline: 0 parsed-value mismatches;
- LightGBM: 293 parsed-value mismatches, 993 serialized-text mismatches, maximum absolute difference `1.0001765041178778e-16`;
- LSTM: 1,124 parsed-value mismatches, 2,938 serialized-text mismatches, maximum absolute difference `1.0061396160665481e-16`;
- learned-model direction changes: 0.

Using only the already-saved canonical predictions and embedded actuals, this audit independently reproduced all four non-regime metric tables with a maximum absolute discrepancy of `8.968520370800093e-16`. It also independently reproduced all saved regime cell, seed-summary, macro, and degradation tables with a maximum absolute discrepancy of `1.8735013540549517e-15`; keys, undefined-value patterns, status fields, contrast orientation, and the low-`n` audit matched. Replacing canonical learned values in memory with the sealed values changes any audited overall metric by at most `1.3322676295501878e-15` and changes no direction.

No additional substantive defect that changes the saved numerical conclusions was found in the bounded checks. That is not proof that no other defect exists. In particular, this audit did not rerun inference to prove checkpoint-to-prediction identity, did not re-access protected 2025 source targets to reconstruct actual returns, and did not refit/recalculate the 2015–2024 threshold distributions from source rows. Threshold provenance is therefore supported by immutable pre-access records, not independently rederived here.

**Portfolio decision: Path A, with strict disclosure.** The F1 artifacts are sufficiently complete and internally reproducible to support explicitly labeled **post-failure exploratory analysis**. They must not be described as a successful pristine, preregistered final test. The independently verified D1–D5 development benchmark and regime study should remain the project's primary scientific evidence; the invalid F1 execution and this forensic analysis are best presented as an engineering/reproducibility case study and a qualified exploratory appendix.

## 2. Original execution status

Final original state: **`INVALID_STOP_NO_RERUN`**.

| Event | UTC time | Evidence |
|---|---|---|
| Atomic execution claim; irreversible consumption point | `2026-09-18T04:45:06.739719Z` | `results/final_test/state/EXECUTION_STARTED.json` |
| Detailed consumed record | `2026-09-18T04:45:06.743705Z` | `results/final_test/state/ONE_TIME_F1_AUTHORIZATION_CONSUMED_b2dbe7a999733df2a8ffc17b1ad6f7f168a7da5d7629b365b23f7a05cb7cd56d.json` |
| Sealed-vector verification completed | `2026-09-18T04:45:08.681972Z` | `results/final_test/verification/sealed_prediction_vector_verification_60959c133a6f2d1c0b6f3b1d97acb2b3e808dbf23bb1e95ad910fffc09a22273.json` |
| Controlled target-source access began | `2026-09-18T04:45:08.683139Z` | `results/final_test/state/ONE_TIME_2025_TARGET_SOURCE_ACCESS_5d961b12d9349b6d3845c00db9c26b5ae5b1aacad43ec987fd01f3dd781ab301.json` |
| Immutable invalid record written | `2026-09-18T04:45:11.336323Z` | `results/final_test/state/FINAL_RUN_INVALID_3ccde4a2b7614a2e0aeb5dc4c12d4ecc871726bc9eb5d1354f796451e1a17a60.json` |

The exact last completed scientific-output stage was the write of `results/final_test/metrics/development_vs_f1_descriptive_comparison_b11013fdcfb902e538fdd2c8f7c8c14e4fb0ab022174e32665200660bbc57a4c.csv`. The subsequent independent verifier completed content-hash checks, authorization-count and prepared-artifact checks, fit-accounting checks, and sealed-vector re-verification. It loaded the canonical raw files, passed the baseline exact prediction comparison, and then failed at the learned-vector equality check. From the preserved iteration order and the independent row audit, LightGBM was the first failing family; the original verifier did not proceed to its later actual-alignment, metric, label, regime, threshold, historical-tree, or source-identity checks.

The comparison occurred **after target access** and after all listed metrics and regime outputs had been written. Those files are preserved failure evidence, not artifacts of a successfully completed F1 protocol.

## 3. Authorization-consumption state

- Authorization SHA-256: `fc920a67fb4462f4a12fbbc4a2f56c3ed8235d079ebc563ac40de57f46302133`.
- Governance-corrected READY SHA-256: `5c3418dd83a7a230828b2bb316002bfa8719f3b1d4eb39c2cd5e53cc96e11fd4`.
- Trust anchor: `FINAL_TEST_GOVERNANCE_V3`.
- Scientific execution ordinal: `1`.
- Atomic claim status: `AUTHORIZATION_IRREVERSIBLY_CONSUMED`.
- Detailed state status: `ONE_TIME_F1_AUTHORIZATION_CONSUMED`.
- Retry after claim: `prohibited`.
- Invalid record says `rerun_prohibited: true`.
- Consumption records found: exactly 1.
- Invalid records found: exactly 1.
- F1 executions during this audit: 0.

The authorization cannot be renewed, reset, or treated as available. This report does not change that state.

## 4. Exact failure location

The preserved traceback identifies `src/evaluation/final_test_evaluation.py`, function `independently_verify_final_artifacts`, at the frozen exact comparison around lines 4177–4184:

1. Reload each sealed file with `pandas.read_csv`.
2. Reload each canonical raw file with `pandas.read_csv`.
3. Sort both by asset/date/model/seed keys.
4. Compare `predicted_log_return` with `numpy.array_equal` and compare direction exactly.
5. Raise `FinalTestError("Canonical predictions differ from their sealed vectors.")` on either failure.

Relevant static implementation path:

- `_sealed_frame` creates finite `float64` prediction vectors.
- `canonical_csv_bytes` writes CSV with LF line endings, `date_format="%Y-%m-%d"`, and `float_format="%.17g"`.
- `generate_and_seal_predictions` writes the three target-free sealed files.
- `verify_sealed_predictions` reloads them with the default pandas parser before target access.
- `build_canonical_raw` copies the already-parsed prediction values and attaches targets by exact `(asset, origin_date, target_date)` keys.
- `write_csv_artifact` serializes the canonical frames again with `%.17g`.
- `independently_verify_final_artifacts` reloads both representations and requires bit-exact array equality.

The exact-equality rule remains controlling. This audit does not replace it with a tolerance and does not retroactively convert the original run to PASS.

## 5. Original and saved artifact inventory

### 5.1 Pre-access and model provenance

| Artifact class | Existing evidence |
|---|---|
| Readiness | `results/final_test/pre_access/READY_FOR_ONE_TIME_2025_ACCESS_GOVERNANCE_CORRECTED_5c3418dd83a7a230828b2bb316002bfa8719f3b1d4eb39c2cd5e53cc96e11fd4.json` |
| Preparation manifest | `results/final_test/pre_access/phase2l_preparation_manifest_25734f61d65b8262a1d7d568c56619149fe3228f010447fc3bafc41797e989c8.json` |
| Prepared references | 81 records; 81 current hashes verified |
| Learned-model files | 56: 16 LightGBM candidates, 4 LightGBM refits, 24 LSTM candidates, 12 LSTM refits |
| Selected structures | `LGBM_03` and `LSTM_21` |
| LSTM seeds | `1729`, `2718`, `31415` |
| Refit audit files | 4 LightGBM rows and 12 LSTM rows; all end at `2024-12-31` and record zero 2025 target rows |

The model directory contains 58 files: the 56 model/checkpoint files plus two refit-audit CSVs. The preparation manifest is the authoritative per-file ledger. All 81 referenced paths, including every model and scaler, exist and match their recorded SHA-256. The refit model hashes also match the consumed-state record. Recorded LightGBM refit iterations equal the saved winning best-iteration table, and all twelve LSTM requested/completed refit epochs equal the saved winning best-epoch table.

This proves preserved provenance and bookkeeping. It does not prove checkpoint-to-prediction identity by fresh inference; inference was prohibited in this audit.

### 5.2 Prediction and target-bearing artifacts

| Family | Sealed target-free rows | Canonical raw rows | Canonical labeled rows |
|---|---:|---:|---:|
| Persistence baseline | 1,000 | 1,000 | 1,000 |
| LightGBM | 1,000 | 1,000 | 1,000 |
| LSTM | 3,000 | 3,000 | 3,000 |

There is no separate saved target-only CSV. The saved actual targets are embedded in each canonical raw and labeled prediction file as `actual_log_return` and `actual_direction`. The target-access record exists, but source targets were not reloaded in this audit.

### 5.3 Saved evaluation artifacts

- Four non-regime metric tables: asset direct, asset LSTM seed summary, macro direct, and macro LSTM seed summary.
- One frozen F1 threshold audit and one threshold-source verification record.
- One 250-row unique 2025 regime-label table.
- Six regime metric tables: baseline, LightGBM, LSTM per seed, LSTM seed summary, macro direct, and macro LSTM seed summary.
- Two degradation tables: asset and macro.
- One low-`n` audit.
- One development-versus-F1 descriptive comparison.

### 5.4 Required completion artifacts that do not exist

- no `phase2l_independent_final_verification` artifact;
- no F1 final run manifest;
- no normal successful Phase 2L execution report.

Their absence is consistent with the preserved failure point.

## 6. Artifact hash verification

### 6.1 Governance, state, and contracts

| Path or identity | SHA-256 |
|---|---|
| `PHASE_2L_B_3_REPORT.md` | `c4395d971e14772842744eb1525a2814adeaab0734cf28bc6d4dee94c9f8c107` |
| `PHASE_2L_B_FINAL_EXECUTION_INVALID_REPORT.md` | `068e8463a496bc2bacd73bc37cc0b5e1e7070568dbbae47bf320ec6d6883ff4e` |
| Preparation manifest | `25734f61d65b8262a1d7d568c56619149fe3228f010447fc3bafc41797e989c8` |
| Governance-corrected READY | `5c3418dd83a7a230828b2bb316002bfa8719f3b1d4eb39c2cd5e53cc96e11fd4` |
| `EXECUTION_STARTED.json` | `e39e9ec420f8c10c978fae1ae9f58220226736f774f25f0b214dff1c68d021f6` |
| Consumed record | `b2dbe7a999733df2a8ffc17b1ad6f7f168a7da5d7629b365b23f7a05cb7cd56d` |
| Sealed-vector verification | `60959c133a6f2d1c0b6f3b1d97acb2b3e808dbf23bb1e95ad910fffc09a22273` |
| Target-access record | `5d961b12d9349b6d3845c00db9c26b5ae5b1aacad43ec987fd01f3dd781ab301` |
| Invalid record | `3ccde4a2b7614a2e0aeb5dc4c12d4ecc871726bc9eb5d1354f796451e1a17a60` |
| `docs/FINAL_TEST_EXECUTION_CONTRACT.md` | `e49c80521b040638b73ce2ce0511c80db328c3068e61aae56f2b87218b2e6f63` |
| `docs/EXPERIMENT_SPEC.md` | `4a29b96a7c826634d2dede63220c378737c0e545a7e733491a0a7002d69fc5b9` |
| `docs/MODEL_PREREGISTRATION.md` | `be04b9ce8457c5b26a8042f548eccce8ddb4b5d1b86a1404a43ef1bb1a52d613` |
| `docs/REGIME_PREREGISTRATION.md` | `2c0a9e9a03cb5f546bc09234ddec147ef720d03f0cddcf88fcb417d0ed341331` |
| `configs/final_test_execution.yaml` | `3c4122ab85fc475a4396d473b146dcd681547ab77cf1a2ab76d9948acd9d86f3` |
| `configs/regime_analysis.yaml` | `e6241685edf87e5e9997dd73e8bf82c34cdfc26ccb92bad2037440925100473d` |
| Certified expanded source tree identity | `45b99fb25ab5e2b6e856489d498e9098d7cf075e27472922afdb863738826900` |

### 6.2 Prediction hashes

| Family | Sealed | Canonical raw | Canonical labeled |
|---|---|---|---|
| Baseline | `bcd174342d815071ce4e3ceaa986c638d772cf8eb8556c77de364498b1a12156` | `dc8799d52a247595b786cc55f249e9b968bc02d029d7e5c5d6b35092c4ef8c5e` | `6ebb05606456a9114024ba2f56d07a5b38e5c81bd0d9adaa9962b64fccb3f6f3` |
| LightGBM | `7059b66335e9850375606cacc8eda2edd7a065e8211aa7f114b2bc195dae0f84` | `4438191f56dfd761c10afc6a062e6539dbc5c5e178fddb330f878b6db5e03a0d` | `0f32de8835024c057bfaa93664fae6797c76bb8900e119be4f20a1d54ff2fea5` |
| LSTM | `3d4c9257c436da912480297f929f74859ea7cd9bf99453459348e63689c6d46c` | `4377da7ccb672100050f3b54b482a37363b8a841528dabdd9cfaf6b5f112346d` | `52f5b6b127eb624aedb8121795928df7b29eb00507c85a1f2cc50934d8599c16` |

### 6.3 Metric and regime hashes

| Artifact | SHA-256 |
|---|---|
| Non-regime asset direct | `8dea87c5314be5bb61bcb00765d56ff8cb55f8500f105fbb46e24d0bb649945a` |
| Non-regime asset LSTM seed summary | `6766312c49e4bf947b65401dbf8b7317384df6780bc6700bfcad520d662a79ee` |
| Non-regime macro direct | `bdf87b913ff2904efa0a4d99e1d24cb201b3a7efc9bcdb4d814f6d671c1d58d3` |
| Non-regime macro LSTM seed summary | `dffa89cacadd5bec7ea306a4f00a65ff02c9bb0c86d1032c164fcdc7506b30e1` |
| F1 threshold audit | `2c5d8525650a0b6031774501c95e637d9c49b0f80577d13f8e0849271139c3b8` |
| F1 threshold-source verification | `69bcb2f6667aa62e7540b1a7e6782db6eced32a315e4d7d3af84e9d05434d30a` |
| Unique F1 regime labels | `6c07fee486542782690d36f48cabe9f8428f2a36eae0cb5f85fd28fb058215f0` |
| Baseline regime cells | `251cbd0e74c5b3e6a554757a91cdf7fd2fd592f1943bea1259a0ae647affdad1` |
| LightGBM regime cells | `3c532030f1cf4b35c8eb71ad4fda70743765999bd2e6cb2c5c6e70367414f88d` |
| LSTM per-seed regime cells | `03f2d15eb6f5f2bbc59420c8bc3107a7137b3bdca003ff2ceff0a028b127b35a` |
| LSTM seed-summary regime cells | `76282e4518aee37fa9b9e4a7aa1a16e934c0d96b38302d1367c0cd5c0031c820` |
| Macro direct regime metrics | `aa1f66689437a7af5e37474d94b1cb10a288ff64726968ce97bb2475ed38872b` |
| Macro LSTM regime summary | `5bac3581396214988f1173a1b355ae80781e298c507abd488fd36aa7ea052a1a` |
| Asset degradation | `e771ce733f42c306b1222724b8526e5c2527f8e4649d99db2408cee4e88d8111` |
| Macro degradation | `bc57c130d2a59de75c09732f7082eaa1238120024c934ab195366a1c91e0ba19` |
| Low-`n` audit | `efcce4159cbbe959e29e159a274c08ecb90cf5e024c845224b92abb31d032e0d` |
| Development-versus-F1 comparison | `b11013fdcfb902e538fdd2c8f7c8c14e4fb0ab022174e32665200660bbc57a4c` |

All 116 content-addressed files currently under `results/final_test/` match the digest in their filenames. The 81 preparation-manifest references also pass individual path/hash verification.

For a broad immutability check, this audit hashed 249 pre-existing files under `data/`, `docs/`, `configs/`, `src/`, `tests/`, and `results/final_test/`, plus the B.3 and INVALID phase reports. The deterministic ledger identity before report creation was `33a27c2d56738cf308eb644d02e25e42788e00c470b7d49016e1102ec0235ba9`.

Post-report evidence check: **PASS**. The same 249 files produced the identical ledger identity `33a27c2d56738cf308eb644d02e25e42788e00c470b7d49016e1102ec0235ba9` after this report was created. No pre-existing evidence, source, test, contract, model, state, prediction, metric, or regime file changed.

## 7. Prediction row and key integrity

Independent checks used the full key `(fold, asset, origin_date, target_date, model, seed)` and the canonical comparison key `(fold, asset, origin_date, target_date)` where model/seed are intentionally varied.

| Check | Result |
|---|---|
| Sealed/raw/labeled row counts | 1,000 / 1,000 / 1,000 baseline; 1,000 / 1,000 / 1,000 LightGBM; 3,000 / 3,000 / 3,000 LSTM |
| Duplicate full keys | 0 in every file |
| Sealed-versus-raw key sets | Exact for every family |
| Sealed-versus-raw row order | Exact after the saved order; no order-only failure |
| Canonical keys | 1,000 unique base keys in every family and every LSTM seed |
| Asset coverage | 250 each for AAPL, GOOGL, MSFT, and NVDA per realization |
| Fold/model/seed identity | Exact: F1; baseline, LightGBM/1729, LSTM/1729/2718/31415 |
| Null cells in prediction artifacts | 0 |
| Non-finite predictions or actuals | 0 |
| Target columns in sealed files | None |
| Learned direction versus prediction sign | 0 mismatches |
| Actual direction versus actual-return sign | 0 mismatches |
| Sealed-versus-canonical direction | 0 mismatches |
| Saved actuals across all models/seeds | Bit-identical on exact keys |
| Origin/target chronology | Every origin precedes target; every target year is 2025 |
| Raw-to-labeled preservation | All non-regime fields preserved exactly after reload |

The baseline direction is intentionally previous-direction persistence rather than the sign of its zero return, as frozen in the contract. The learned-model directions are strictly the sign of their continuous forecasts.

The saved actuals are internally aligned and use the documented schema. The processed-data manifest and target implementation define `actual_log_return` as the next observed asset-local session `ln(close[target_date] / close[origin_date])`, with direction 1 iff the value is positive; the raw-data manifest records `auto_adjust: true`. This audit did not independently recalculate the 2025 returns from source prices, so source-level target correctness remains a limitation rather than a newly verified fact.

## 8. Numerical discrepancy analysis

The verifier's default-parser comparison produced:

| Family | Rows | Serialized-text mismatches | Parsed bit mismatches | Max absolute difference | Max relative difference | Median / max ULP difference | Direction mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1,000 | 0 | 0 | `0` | n/a | `0 / 0` | 0 |
| LightGBM | 1,000 | 993 | 293 | `1.0001765041178778e-16` | `8.52534473669878e-13` | `461 / 7,378` | 0 |
| LSTM | 3,000 | 2,938 | 1,124 | `1.0061396160665481e-16` | `8.654481085556646e-13` | `230 / 7,378` | 0 |

ULP counts are technically valid for same-sign paired `float64` values, but the maximum ULP count occurs near `1.2e-4`; the absolute difference remains about `1e-16`. No pair crosses zero.

Representative maximum-relative-difference rows:

- LightGBM, GOOGL, origin `2025-01-07`, target `2025-01-08`, seed `1729`: sealed text `-0.00011728621943823293`; canonical text `-0.00011728621943819999`; default-parser values differ by `9.999054535747565e-17`, relative difference `8.52534473669878e-13`, ULP difference 7,378.
- LSTM, AAPL, origin `2025-09-08`, target `2025-09-09`, seed `31415`: sealed text `-0.0001155361533164978`; canonical text `-0.00011553615331639999`; default-parser values differ by `9.999054535747565e-17`, relative difference `8.654481085556646e-13`, ULP difference 7,378.

These small differences do not satisfy exact equality. Their small magnitude does not excuse the original failure, but it is relevant to whether the saved numerical conclusions materially change.

## 9. Serialization-root-cause assessment

**Assessment: CONFIRMED.**

The following independent observations jointly identify the mechanism:

1. For baseline, LightGBM, and LSTM, canonical text equals `format(default_parser(sealed_value), ".17g")` on 1,000/1,000, 1,000/1,000, and 3,000/3,000 rows respectively.
2. Parsing canonical text with pandas `float_precision="round_trip"` recovers the default-parser sealed value bit-for-bit on every row.
3. Re-reading that canonical text with the default parser changes 293 LightGBM values and 1,124 LSTM values relative to the intermediate value, exactly the counts that fail the frozen verifier.
4. Round-trip parsing the original sealed and canonical strings shows 993 LightGBM and 2,938 LSTM encoded-float differences, matching the serialized-text mismatch counts.
5. Raw and labeled prediction text is identical on every row, so regime labeling did not introduce another prediction change.
6. Baseline zero values survive both cycles exactly and do not fail.

The root cause is therefore more precise than “CSV rounding”: a default CSV parse changed many originally serialized learned floats, the canonical writer faithfully serialized those intermediate values, and a second default parse changed a subset again. The exact comparator then compared values from different parse generations.

This defect also means last-place numeric identity should not be inferred across other CSV reload boundaries. However, the independent metric reproductions below show no direction, sign, sample-status, ranking, or reported conclusion change at the observed scale.

## 10. Additional-defect audit

| Domain | Result | Evidence and limit |
|---|---|---|
| Prediction alignment and coverage | **VERIFIED** | Exact keys, rows, assets, dates, seeds, models, finite values, and chronology pass. |
| Model/checkpoint provenance | **SUPPORTED WITH LIMITATIONS** | 81/81 manifest records and every model hash pass; selections and refit budgets agree. No inference replay was allowed. |
| Actual-target alignment | **VERIFIED within saved artifacts** | Actual returns/directions are bit-identical across all families/seeds on exact keys. Source prices were not reopened. |
| Target units and direction | **SUPPORTED WITH LIMITATIONS** | Frozen contract, source code, and processed manifest define next-session log return; every saved direction matches `actual_log_return > 0`. Source-level return reconstruction was not performed. |
| Metric definitions | **VERIFIED** | Independent MAE, RMSE, DA, class balance, matched baseline, MAE Skill, DA difference, equal-asset macro, and seed-first summaries reproduce. |
| Regime threshold provenance | **SUPPORTED WITH LIMITATIONS** | Pre-access threshold artifact says 2015–2024 only; READY and preparation manifest bind its hash; saved source-verification says no 2025 distribution. Source rows were not reloaded to refit/recalculate thresholds. |
| Regime labels and joins | **VERIFIED** | 250 unique origins; all rules reproduce; exact many-to-one joins retain all prediction rows and labels across families/seeds. |
| Sample and aggregation rules | **VERIFIED** | `n >= 30`, per-seed-first LSTM calculations, equal-asset macro values, contrast orientation, and low-`n` audit reproduce. |

**Additional substantive defects: NOT FOUND in the bounded audit.** This statement means no second conclusion-changing defect was identified in the permitted evidence; it does not assert exhaustive absence.

The missing final verification, run manifest, and normal report are consequential completion gaps caused by the original stop. They are not evidence that the saved tables passed the original verification.

## 11. Target and metric reproducibility

### 11.1 Target evidence

All 1,000 base keys carry one finite saved actual return and direction. The values are identical across baseline, LightGBM, and each LSTM seed, and all 5,000 canonical rows have direction equal to `actual_log_return > 0`. The date pairs are strictly chronological and target dates span the 250 saved 2025 sessions per asset.

No independent target-only artifact exists. The target values were not compared with protected source rows or reconstructed from prices during this audit. Therefore:

- saved-target alignment: **VERIFIED**;
- target sign consistency: **VERIFIED**;
- source-to-saved target identity: **NOT VERIFIED in this audit**.

### 11.2 Independent reproduction of saved non-regime metrics

Maximum absolute saved-versus-recomputed discrepancy:

| Table | Rows | Key match | Undefined-value pattern | Maximum absolute discrepancy |
|---|---:|---|---|---:|
| Asset direct | 20 | exact | exact | `8.968520370800093e-16` |
| Asset LSTM seed summary | 4 | exact | exact | `7.077671781985373e-16` |
| Macro direct | 5 | exact | exact | `5.273559366969494e-16` |
| Macro LSTM seed summary | 1 | exact | exact | `5.273559366969494e-16` |

These are CSV reload/representation-scale differences and are far below the original verifier's later metric tolerance of `1e-12`.

Selected independently reproduced equal-asset F1 values, which are **post-failure audit results**, not a valid final-test PASS:

| Model | MAE | RMSE | DA | MAE Skill vs zero-return baseline | DA difference vs direction persistence |
|---|---:|---:|---:|---:|---:|
| Persistence baseline | `0.014869915629502932` | `0.02176945271641622` | `0.478` | n/a | n/a |
| LightGBM | `0.014821168884570853` | `0.021804041335958535` | `0.530` | `0.002477400303526389` | `0.05200000000000002` |
| LSTM seed mean | `0.015237191286519658` | `0.022062339019920496` | `0.5133333333333333` | `-0.02841156731966385` | `0.035333333333333335` |

LightGBM asset MAE Skill is positive for AAPL (`0.0011019821`), MSFT (`0.0023050493`), and NVDA (`0.0088393876`), and negative for GOOGL (`-0.0023368177`). LSTM seed-mean macro MAE Skill is negative; no best-seed substitution is permitted.

Using sealed rather than canonical learned returns while retaining the already-saved actuals changes any audited overall metric by at most `1.3322676295501878e-15`; direction metrics do not change. Thus the serialization defect is numerically immaterial to these saved overall conclusions, while still invalidating the original exact verification.

## 12. Regime reproducibility

### 12.1 Threshold provenance

The frozen pre-access threshold audit records:

- permitted target dates: `2015-01-01` through `2024-12-31`;
- latest contributing target year: 2024;
- unique permitted origins: 2,515;
- finite annualized RV21 values: 2,494;
- volatility median: `0.11953373036311318`;
- finite absolute-return values: 2,514;
- linear-interpolation q95 shock threshold: `0.021654533903868428`;
- post-2024 rows included: 0;
- source unit: one unique SPY observation per origin date.

The separately saved threshold-source verification reports 10,060 repeated asset rows, 2,515 unique origins, and `2025_distribution_used: false`. The threshold audit existed pre-access, is referenced by the preparation manifest, and is bound into the signed READY and consumed record.

This audit verified that provenance chain and the saved metadata. It did not reload pre-2025 source rows and recompute the median/q95, because the audit was restricted from scientific-pipeline execution and additional protected-data access. Threshold numerical provenance is therefore **supported with limitations**, not newly independently fitted or verified from source.

### 12.2 Label integrity

The unique-label artifact has 250 rows and 250 unique origins, with no duplicates:

| Dimension | Saved counts |
|---|---|
| Trend | 64 negative, 186 positive |
| Volatility | 147 high, 103 low |
| Transition | 67 transition, 183 stable |

Recalculation directly from the saved label-source columns found zero rule mismatches for trend, volatility, or transition. Every row records latest threshold year 2024 and `threshold_refit_on_2025 = false`. Label joins are complete for all 1,000 baseline, 1,000 LightGBM, and 3,000 LSTM rows; joined target dates match, and there are zero label disagreements across assets, models, or seeds.

### 12.3 Saved regime-table reproduction

| Table | Rows | Keys/status match | Maximum absolute discrepancy |
|---|---:|---|---:|
| Baseline cells | 24 | exact | `1.1102230246251565e-16` |
| LightGBM cells | 24 | exact | `8.149947730573537e-16` |
| LSTM per-seed cells | 72 | exact | `1.8735013540549517e-15` |
| LSTM seed-summary cells | 24 | exact | `1.4155343563970746e-15` |
| Macro direct cells | 30 | exact | `5.507747036226363e-16` |
| Macro LSTM seed summary | 6 | exact | `5.507747036226363e-16` |
| Asset degradation | 24 | exact | `1.0704328048949385e-15` |
| Macro degradation | 6 | exact | `3.421742056364252e-16` |

The independently rebuilt low-`n` audit matches the saved file exactly. All 24 baseline, 24 LightGBM, 72 LSTM per-seed, and 24 LSTM seed-summary cells are reportable; all 24 asset and 6 macro degradation contrasts are reportable under the frozen `n >= 30` rule.

Selected reproduced macro degradations, again **post-failure exploratory**:

| Model | Stress contrast | MAE Skill degradation | DA degradation |
|---|---|---:|---:|
| LightGBM | transition − stable | `-0.0029703114699924926` | `-0.025691216050893084` |
| LightGBM | high − low volatility | `-0.002115952439915858` | `-0.01089756290865862` |
| LightGBM | negative − positive trend | `0.004538054854376555` | `-0.0035702284946236285` |
| LSTM seed mean | transition − stable | `0.00009133408101039825` | `0.007272381263083494` |
| LSTM seed mean | high − low volatility | `-0.04239600923814069` | `-0.03098650463421604` |
| LSTM seed mean | negative − positive trend | `-0.028900402587832903` | `-0.01967125896057348` |

Downstream saved regime results are reproducible from saved labels and predictions. End-to-end regime reproducibility is **partial** only because threshold fitting was not independently recalculated from source history in this bounded audit.

## 13. Verified findings

| Scientific-validity question | Status | Answer and evidence |
|---|---|---|
| Q1. Was exact sealed-versus-canonical equality violated? | **VERIFIED** | Yes: 293 LightGBM and 1,124 LSTM default-parser bit mismatches; baseline exact. |
| Q2. Is CSV serialization/parsing a supported explanation? | **VERIFIED** | Yes; the complete parse → `.17g` write → parse chain reproduces every canonical string and every failing count. |
| Q3. Is serialization the only defect identified? | **SUPPORTED WITH LIMITATIONS** | It is the only substantive defect found. Bounded checks cannot prove no undiscovered defect exists. |
| Q4. Were saved predictions themselves independently verified? | **SUPPORTED WITH LIMITATIONS** | Target-free sealed hashes/schema/keys were verified before target access and again here. No checkpoint inference replay occurred; canonical exact identity failed. |
| Q5. Are existing overall metrics reproducible? | **VERIFIED** | Yes, from saved canonical rows, with maximum discrepancy `8.97e-16`. |
| Q6. Are existing regime metrics reproducible? | **SUPPORTED WITH LIMITATIONS** | Labels, joins, metrics, seed/macro layers, degradation, and low-`n` status reproduce. Threshold fitting was not rederived from source. |
| Q7. Does an identified defect materially affect numerical conclusions? | **VERIFIED** | No identified defect changes direction, status, ranking, or audited conclusions; the exact-equality defect nevertheless invalidates the run procedurally. |
| Q8. Which findings have independent support? | **VERIFIED** | Row/key integrity, saved-target alignment, parse mechanism, overall metrics, label rules/joins, regime metrics/degradation, low-`n`, and aggregation rules. |
| Q9. Which findings remain uncertain? | **NOT VERIFIED** | Fresh checkpoint-to-prediction identity, source-price-to-saved-target identity, and source-history-to-threshold recalculation. |
| Q10. Which public claims must be excluded? | **VERIFIED** | Any successful/pristine final-test claim, confirmatory F1 claim, exact prediction-identity claim, causal/profitability/significance claim, or claim that 2025 was untouched. |

## 14. Unresolved findings

1. **Checkpoint-to-prediction identity.** Hash and metadata provenance is strong, but no model inference was rerun. The sealed vectors cannot be newly proven to be the outputs of those checkpoints under this audit's restrictions.
2. **Source-level target identity.** Saved actuals align perfectly across artifacts, but no independent target-only file exists and protected 2025 source rows were not reopened.
3. **Threshold recalculation.** The pre-access threshold artifact and provenance chain are intact, but the 2015–2024 source distribution was not independently reprocessed here.
4. **Unexecuted tail of the original verifier.** The original process never reached its later metric/regime/threshold/source-tree verification stages. This audit reproduced much of that logic independently, but it is not the missing original PASS artifact.
5. **Exhaustiveness.** “No additional defect found” is evidence-bounded, not proof of absence.
6. **Commit provenance.** The project has relied on content hashes because usable Git revision metadata was unavailable in the certified snapshot.

## 15. Scientific limitations

- The original F1 execution is permanently invalid and cannot be described as a successful one-time test.
- Historical integration tests had already accessed 2025 features and constructed/materialized 2025 targets. No evidence of earlier performance evaluation or methodology contamination was found, but 2025 was not pristine or untouched.
- The post-failure audit necessarily examined saved 2025 outcomes after the original stop. Its calculations are retrospective and exploratory.
- F1 covers one calendar year and four large US equities; it does not establish universal generalization.
- Daily rows are serially dependent; no IID significance test or confidence claim is supported.
- Regime dimensions overlap and are descriptive associations, not causal mechanisms.
- Directional accuracy does not establish profitability, tradability, calibration, or economic value.
- PatchTST is absent from F1 and must not be implied to have participated.
- Exact numerical values in this report come from immutable saved files, but target-source and checkpoint replay limitations remain as described above.

## 16. Publication-safe claims

The following wording is defensible if the disclosure appears with the result rather than in a remote footnote:

- “The original one-time 2025 execution was invalidated by an exact saved-file identity failure. A later read-only forensic audit found a deterministic CSV parse-and-reserialize mechanism and no direction changes.”
- “Post-failure exploratory calculations from preserved artifacts reproduced the saved overall and regime tables to approximately `2e-15`, but do not constitute a successful preregistered final test.”
- “In the preserved 2025 artifacts, LightGBM had equal-asset MAE Skill of about `+0.25%` relative to zero-return persistence, while the LSTM seed mean was about `-2.84%`; these values are exploratory.”
- “In the preserved 2025 regime artifacts, the LSTM's baseline-relative MAE Skill was lower in high- versus low-volatility observations and in negative- versus positive-trend observations; this is descriptive, not causal.”
- “The project's primary verified evidence remains the D1–D5 2020–2024 walk-forward study: LightGBM produced small, fold-dependent MAE gains in 3/5 macro folds, while the LSTM seed mean did not beat zero-return persistence on macro MAE in any development fold.”
- “The verified development regime study found heterogeneous stress patterns, incomplete five-fold macro support, and 15/88 reportable asset/fold contrasts where MAE-Skill and directional conclusions disagreed.”
- “The invalid run is itself an engineering result: content-addressing alone did not guarantee cross-generation numeric identity under the chosen CSV parser.”

All F1 claims should use labels such as **post-failure**, **exploratory**, **preserved-artifact**, and **original run invalid**.

## 17. Claims that must be excluded

Do not claim that:

- F1/2025 passed, validated, confirmed, replicated, or successfully completed the preregistered final test;
- sealed and canonical learned predictions were exactly identical;
- the saved F1 metrics passed the original independent verifier;
- the final run manifest or final independent-verification artifact exists;
- fresh inference proved that the saved predictions came from the recorded checkpoints;
- this audit independently reconstructed 2025 targets from prices or refit the 2015–2024 thresholds;
- F1 confirms a D1–D5 pattern;
- 2025 was never previously accessed;
- a regime caused a model failure;
- any result is statistically significant, universally generalizable, profitable, or suitable for trading;
- no other defect can exist;
- PatchTST has an F1 result.

## 18. Graduate-application portfolio decision

### Path A — recommended

Use the verified D1–D5 walk-forward and development regime results as the primary study. Include the invalid F1 episode as a transparent reproducibility case study. If F1 numerical findings are shown, place them in a distinct section titled, for example, “Post-Failure Exploratory Analysis of Preserved 2025 Artifacts,” and repeat the invalid-status disclosure in that section.

Path A is feasible because the saved population is complete, targets align internally, overall metrics reproduce, regime rules/joins/metrics reproduce, the serialization mechanism is confirmed, and no conclusion-changing numerical defect was found. Its cost is that F1 cannot serve as confirmatory evidence.

### Path B — defensible fallback

Exclude all F1 numerical conclusions and finish the portfolio using only already-verified development evidence. That evidence supports, among other points:

- LightGBM had positive macro MAE Skill in D1, D4, and D5, but negative skill in D2 and D3; there was no universal winner.
- LSTM seed-mean macro MAE Skill was negative in all five development folds and negative in 16/20 asset/fold cells.
- Model ranking varied across folds, and added complexity did not produce stable monotonic improvement over persistence.
- In reportable development asset/fold contrasts, worse stressed-regime MAE Skill occurred in 4/16 LightGBM transition, 7/16 LightGBM volatility, 2/12 LightGBM trend, 8/16 LSTM transition, 9/16 LSTM volatility, and 2/12 LSTM trend contrasts.
- Thirty-two asset/fold and eight macro regime contrasts were suppressed from headline use by the frozen sample-size rule.
- Fifteen of 88 reportable asset/fold regime contrasts had opposing MAE-Skill and direction-degradation conclusions.

Path B sacrifices the preserved 2025 descriptive evidence but remains scientifically complete enough for a graduate-school CS portfolio centered on leakage-aware evaluation, baseline discipline, model instability, regime heterogeneity, and reproducibility engineering.

### Decision

**Recommend Path A**, provided the invalid status and exploratory nature are prominent and inseparable from the F1 results. If that disclosure cannot be maintained in the intended format, use Path B. Do not run a new F1 experiment and do not rebuild the trust-anchor system.

## 19. Exact remaining work required to finish the project

No scientific re-execution is required or recommended. After review of this report:

1. Choose Path A or Path B explicitly.
2. Freeze a short claims ledger using Sections 16 and 17.
3. Build the portfolio narrative around the already-verified D1–D5 results; if Path A is chosen, isolate F1 in a clearly labeled post-failure exploratory appendix.
4. Add only presentation-layer tables/figures derived from the audited immutable artifacts, with provenance and invalid-status captions; do not replace canonical artifacts.
5. Update the public README/report/presentation only after the scientific wording is approved.
6. Preserve the atomic claim, consumption record, target-access record, INVALID marker, sealed/canonical files, and this audit report permanently.
7. Optionally obtain an external human review of the disclosure and claims ledger. This is review, not authorization for another F1 run.

This report is the only project artifact created by the final forensic audit.
