# Phase 2G — Combined Development Benchmark Audit and PatchTST Gate Decision

Project: **When Stock Forecasting Models Fail: Stress-Testing Robustness Under Market Regime Shifts**  
Subtitle: **A Leakage-Aware Walk-Forward Evaluation of Classical and Deep Learning Models**  
Experiment specification: **version 1.1**  
Audit generation time: **2026-09-14T15:00:42.828778Z**

## 1. Phase status

**Phase 2G completed successfully.** The frozen Phase 2C baseline, Phase 2E LightGBM, and Phase 2F LSTM results passed integrity, scope, exact-key, actual-value, and independent metric-reproduction checks. The audit created new content-addressed comparison artifacts only under `results/combined/`.

No benchmark model was fitted or refitted. No model configuration, feature, target, seed, fold, historical prediction, or historical metric was changed. PatchTST was not implemented or installed. Regime analysis was not performed, and F1/2025 performance remained untouched.

The explicit G1–G7 decision is:

> **ADMIT PATCHTST TO A PREREGISTERED PILOT**

This admits only a separately preregistered, bounded pilot. It does not authorize a full PatchTST benchmark.

## 2. Frozen artifact integrity

Every authoritative file below matched its SHA-256 filename/identity, its phase report, and its run-manifest binding.

| Phase | Artifact | SHA-256 | Status |
|---|---|---|---|
| 2C | Baseline predictions | `e09e4909e9af7b7836c5e0b2c8d7b320d2848d5da9b33b8c9492421e0a108bb4` | PASS |
| 2C | Baseline asset metrics | `ac1a32f81a6cf597e186f317ca2983ac15c0cf45c75c2c0968a4dce85e6a6a16` | PASS |
| 2C | Baseline macro metrics | `27bb8779cf550b7cf288e098f7551eac43100879afe09786265ef70d69be65c8` | PASS |
| 2C | Baseline manifest | `9927474b9a3ef014ad4e016c48a7f8cda461a0ccb3e0daaab6f124135ddc32d0` | PASS |
| 2E | LightGBM predictions | `36445836777cba815bc240c0c7229838cdc2177e94f50bac7905b8e2a9ace0c7` | PASS |
| 2E | LightGBM asset metrics | `89aa75ff8d6833eb9f0089c94126307fd94bdf93899375bcb6b5616c86e5a5e5` | PASS |
| 2E | LightGBM macro metrics | `ce1ad407b42129b78f2c654f9c797aecb22ea916f6ee3e9ba229efcd386a879e` | PASS |
| 2E | LightGBM manifest | `6b90c0907a42933b3d184bc87a6bc720dbf846a9d751e475c2657539e815bf75` | PASS |
| 2F | LSTM predictions | `f624bc00c41a430c6e79188fe8b10a1bc03714971719789d5162446c70c7df60` | PASS |
| 2F | LSTM per-seed metrics | `666a6caeae13744df79919e2f30466c9c5a025c5eb2966e785f90c0469159ede` | PASS |
| 2F | LSTM asset/seed summary | `e2d2cae825530d985d6bac671290f0a1ccd2a250043fb3617cfbfc3c4f47aa6a` | PASS |
| 2F | LSTM macro by seed | `f833635ae0e159f5d797886a10aa9136ece637eaaa0091d854e592b000a9eb33` | PASS |
| 2F | LSTM macro seed summary | `d0ad5cc9b434e53e856f60ff03b37f6d184aa40b98b81da15540e19750148a17` | PASS |
| 2F | LSTM manifest | `a657eaa1426e369ff68275c75bac0e4e8c4e91f33fefde8d7849c51eb85df283` | PASS |

Full-tree identities were captured before the audit and rechecked after all combined artifacts were written:

| Frozen tree | Files | Aggregate SHA-256 before | Aggregate SHA-256 after |
|---|---:|---|---|
| `results/baselines/` | 6 | `1f42bb6758b54251060983e1b7a9323861d94ac768ee5371571e77e5b46fc466` | identical |
| `results/lightgbm/` | 113 | `6f1079692067a8ed49dcce3d3db52efd22e00883084575a3a804a6f942e9d970` | identical |
| `results/lstm/` | 235 | `64d0866469259062e49652c17be364e34cde6bbe7fecc17dbb821489e6b66f61` | identical |

The experiment specification SHA-256 was `4a29b96a7c826634d2dede63220c378737c0e545a7e733491a0a7002d69fc5b9`; the learned-model preregistration SHA-256 was `be04b9ce8457c5b26a8042f548eccce8ddb4b5d1b86a1404a43ef1bb1a52d613`.

## 3. Canonical key audit

The canonical key was exactly `(fold, asset, origin_date, target_date)`. No intersection, inner-join shrinkage, omission, or positional alignment was used.

| Artifact/slice | Rows | Unique canonical keys | Result |
|---|---:|---:|---|
| Baseline | 5,032 | 5,032 | PASS |
| LightGBM | 5,032 | 5,032 | PASS |
| LSTM seed 1729 | 5,032 | 5,032 | PASS |
| LSTM seed 2718 | 5,032 | 5,032 | PASS |
| LSTM seed 31415 | 5,032 | 5,032 | PASS |
| LSTM total | 15,096 | 5,032 per seed | PASS |

All key sets were exactly equal. There were no missing, extra, or duplicate keys. On every matched row, `actual_log_return` had maximum absolute difference **0.0**, and `actual_direction` matched exactly.

## 4. Independent metric recomputation

All metrics were recomputed from the three canonical row-level prediction files, not copied from phase reports. MAE, RMSE, directional accuracy, positive balance, zero-return baseline MAE, MAE Skill, direction-persistence accuracy, and DA difference were reconstructed at their required reporting levels.

| Saved table checked | Maximum absolute numerical difference |
|---|---:|
| Baseline asset/fold | `1.1102230246251565e-16` |
| Baseline macro fold | `1.1102230246251565e-16` |
| LightGBM asset/fold | `1.4710455076283324e-15` |
| LightGBM macro fold | `4.779163176316104e-16` |
| LSTM asset/fold/seed | `7.216449660063518e-16` |
| LSTM asset/fold seed summary | `7.216449660063518e-16` |
| LSTM macro by seed | `3.3306690738754696e-16` |
| LSTM macro seed summary | `2.7755575615628914e-16` |

The largest discrepancy was `1.48e-15`, attributable to floating-point serialization/recalculation and far below the `1e-12` verification tolerance. Macro metrics used equal asset weighting. LSTM seed means were means of separately computed seed metrics; row-level predictions were never averaged.

## 5. Baseline summary

The baseline combines a zero next-day log-return forecast for MAE/RMSE with previous-direction persistence for directional accuracy.

| Fold | Observations | Macro MAE | Macro RMSE | Direction-persistence DA |
|---|---:|---:|---:|---:|
| D1 | 1,012 | 0.020461 | 0.029525 | 0.425889 |
| D2 | 1,008 | 0.013673 | 0.018191 | 0.496032 |
| D3 | 1,004 | 0.021334 | 0.027220 | 0.521912 |
| D4 | 1,000 | 0.014237 | 0.019414 | 0.519000 |
| D5 | 1,008 | 0.014357 | 0.019369 | 0.518849 |

Baseline skill against itself remains zero by definition and is not evidence of predictive value.

## 6. LightGBM development summary

Fold winners were LGBM_01 for D1, LGBM_02 for D2, LGBM_01 for D3, LGBM_03 for D4, and LGBM_01 for D5. Selection remained historical and unchanged.

LightGBM had positive macro MAE Skill in **3/5** folds: D1 (`+0.005890`), D4 (`+0.006915`), and D5 (`+0.006465`). It was negative in D2 (`-0.002023`) and D3 (`-0.017863`). Across the 20 asset/fold cells, MAE Skill was positive in **12**, zero in **0**, and negative in **8**.

Directionally, LightGBM exceeded direction persistence in **4/5** macro folds and in **14/20** asset/fold cells. The notable macro failure was D3, where MAE Skill was `-0.017863` and DA difference was `-0.050797`.

The evidence is modest and inconsistent rather than a broad, stable improvement over persistence.

## 7. LSTM development summary

Fold winners remained LSTM_63 for D1, D2, and D4, and LSTM_21 for D3 and D5. All three seeds—1729, 2718, and 31415—remain separate.

LSTM seed-mean macro MAE Skill was negative in every fold:

- D1: `-0.067565`
- D2: `-0.024038`
- D3: `-0.057713`
- D4: `-0.021335`
- D5: `-0.021775`

At asset/fold level, the three-seed mean MAE Skill was positive in **4/20**, zero in **0/20**, and negative in **16/20** cells. The macro result therefore provides no development evidence that this preregistered LSTM improves return-error accuracy over zero-return persistence.

Directional results were mixed: seed-mean DA difference was positive in D1, D2, and D4, negative in D3 and D5, and positive in **11/20** asset/fold cells. Better sign classification in some cells did not translate into lower MAE.

## 8. Combined fold/asset comparison

The canonical 20-row table contains the full baseline MAE/RMSE/DA fields, LightGBM fields, LSTM seed-mean/seed-dispersion fields, and selected configuration context. The compact view below reports the two baseline-relative contrasts.

| Fold | Asset | LGBM config | LGBM MAE Skill | LGBM DA diff | LSTM context | LSTM seed-mean MAE Skill | LSTM seed-mean DA diff |
|---|---|---|---:|---:|---|---:|---:|
| D1 | AAPL | LGBM_01 | 0.002903 | 0.130435 | LSTM_63 | -0.022979 | 0.111989 |
| D1 | GOOGL | LGBM_01 | 0.008171 | 0.090909 | LSTM_63 | -0.080528 | 0.076416 |
| D1 | MSFT | LGBM_01 | 0.001064 | 0.134387 | LSTM_63 | -0.180780 | 0.090909 |
| D1 | NVDA | LGBM_01 | 0.011421 | 0.126482 | LSTM_63 | 0.014028 | 0.088274 |
| D2 | AAPL | LGBM_02 | -0.003355 | 0.027778 | LSTM_63 | -0.026030 | 0.037037 |
| D2 | GOOGL | LGBM_02 | 0.013808 | 0.067460 | LSTM_63 | 0.008991 | 0.033069 |
| D2 | MSFT | LGBM_02 | -0.006574 | 0.011905 | LSTM_63 | -0.051189 | 0.023810 |
| D2 | NVDA | LGBM_02 | -0.011973 | 0.011905 | LSTM_63 | -0.027923 | -0.002646 |
| D3 | AAPL | LGBM_01 | -0.047858 | -0.019920 | LSTM_21 | -0.060951 | -0.033201 |
| D3 | GOOGL | LGBM_01 | -0.012624 | -0.067729 | LSTM_21 | -0.033630 | -0.027888 |
| D3 | MSFT | LGBM_01 | -0.009243 | -0.043825 | LSTM_21 | -0.102535 | -0.010624 |
| D3 | NVDA | LGBM_01 | -0.001728 | -0.071713 | LSTM_21 | -0.033737 | -0.066401 |
| D4 | AAPL | LGBM_03 | 0.006840 | 0.060000 | LSTM_63 | -0.031975 | 0.024000 |
| D4 | GOOGL | LGBM_03 | 0.004234 | 0.040000 | LSTM_63 | 0.002234 | 0.050667 |
| D4 | MSFT | LGBM_03 | 0.005671 | 0.036000 | LSTM_63 | -0.007461 | 0.020000 |
| D4 | NVDA | LGBM_03 | 0.010916 | -0.008000 | LSTM_63 | -0.048139 | -0.082667 |
| D5 | AAPL | LGBM_01 | 0.004786 | 0.031746 | LSTM_21 | -0.028051 | -0.033069 |
| D5 | GOOGL | LGBM_01 | 0.023846 | 0.059524 | LSTM_21 | -0.001691 | -0.007937 |
| D5 | MSFT | LGBM_01 | -0.010003 | -0.015873 | LSTM_21 | -0.057811 | -0.027778 |
| D5 | NVDA | LGBM_01 | 0.007229 | 0.063492 | LSTM_21 | 0.000455 | 0.055556 |

Full-precision artifact: `results/combined/development_asset_fold_comparison_61ad31f5e745338ccf679be9614188bab869964b350ab4f32a26067f85a2f3a1.csv`.

## 9. Macro fold comparison

| Fold | Baseline MAE | LGBM MAE | LGBM Skill | LGBM DA diff | LSTM seed-mean MAE | LSTM Skill | LSTM DA diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 0.020461 | 0.020333 | 0.005890 | 0.120553 | 0.021668 | -0.067565 | 0.091897 |
| D2 | 0.013673 | 0.013726 | -0.002023 | 0.029762 | 0.014006 | -0.024038 | 0.022817 |
| D3 | 0.021334 | 0.021655 | -0.017863 | -0.050797 | 0.022467 | -0.057713 | -0.034529 |
| D4 | 0.014237 | 0.014131 | 0.006915 | 0.032000 | 0.014585 | -0.021335 | 0.003000 |
| D5 | 0.014357 | 0.014248 | 0.006465 | 0.034722 | 0.014570 | -0.021775 | -0.003307 |

Equal asset weighting was used for every macro field. LSTM values are seed-summary metrics, not scores of averaged forecasts.

Primary MAE ranks by fold were:

| Fold | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| D1 | LightGBM | Baseline | LSTM seed-mean |
| D2 | Baseline | LightGBM | LSTM seed-mean |
| D3 | Baseline | LightGBM | LSTM seed-mean |
| D4 | LightGBM | Baseline | LSTM seed-mean |
| D5 | LightGBM | Baseline | LSTM seed-mean |

The ranking changes across folds; there is no universal development winner. DA was ranked separately and was not combined with MAE into a synthetic score.

## 10. Cross-fold descriptive statistics

These are descriptive summaries of five fold values, not confidence intervals or formal inference.

| Model | Fold-level metric | Median | Q1 | Q3 | IQR | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | Macro MAE Skill | 0.005890 | -0.002023 | 0.006465 | 0.008488 | -0.017863 | 0.006915 |
| LightGBM | Macro DA difference | 0.032000 | 0.029762 | 0.034722 | 0.004960 | -0.050797 | 0.120553 |
| LightGBM | Macro MAE | 0.014248 | 0.014131 | 0.020333 | 0.006202 | 0.013726 | 0.021655 |
| LightGBM | Macro DA | 0.546443 | 0.525794 | 0.551000 | 0.025206 | 0.471116 | 0.553571 |
| LSTM seed-mean | Macro MAE Skill | -0.024038 | -0.057713 | -0.021775 | 0.035939 | -0.067565 | -0.021335 |
| LSTM seed-mean | Macro DA difference | 0.003000 | -0.003307 | 0.022817 | 0.026124 | -0.034529 | 0.091897 |
| LSTM seed-mean | Macro MAE | 0.014585 | 0.014570 | 0.021668 | 0.007098 | 0.014006 | 0.022467 |
| LSTM seed-mean | Macro DA | 0.517787 | 0.515542 | 0.518849 | 0.003307 | 0.487384 | 0.522000 |

No ordinary IID test was run across daily observations.

## 11. Model-versus-baseline win/loss counts

Zero means absolute value at or below the strict `1e-12` tolerance.

| Model/scope | Metric | Positive | Zero | Negative | Total |
|---|---|---:|---:|---:|---:|
| LightGBM, 20 asset/fold cells | MAE Skill | 12 | 0 | 8 | 20 |
| LightGBM, 20 asset/fold cells | DA difference | 14 | 0 | 6 | 20 |
| LSTM three-seed metric mean | MAE Skill | 4 | 0 | 16 | 20 |
| LSTM three-seed metric mean | DA difference | 11 | 0 | 9 | 20 |
| LSTM seed 1729 | MAE Skill | 3 | 0 | 17 | 20 |
| LSTM seed 1729 | DA difference | 12 | 0 | 8 | 20 |
| LSTM seed 2718 | MAE Skill | 8 | 0 | 12 | 20 |
| LSTM seed 2718 | DA difference | 12 | 1 | 7 | 20 |
| LSTM seed 31415 | MAE Skill | 3 | 0 | 17 | 20 |
| LSTM seed 31415 | DA difference | 14 | 0 | 6 | 20 |

The primary LSTM comparison is the three-seed metric mean. The per-seed rows are descriptive robustness evidence only; seed 2718 was not promoted or selected because it had more positive cells.

## 12. Directional comparison

Regression-error and sign performance often disagreed:

- LightGBM had opposite MAE-Skill/DA-difference signs in **4/20** cells: D2 AAPL, D2 MSFT, D2 NVDA, and D4 NVDA.
- LSTM seed-mean had opposite signs in **7/20** cells: D1 AAPL/GOOGL/MSFT, D2 AAPL/MSFT, and D4 AAPL/MSFT.
- LightGBM's strongest macro DA gain was D1 (`+0.120553`); its largest macro DA loss was D3 (`-0.050797`).
- LSTM's strongest macro DA gain was D1 (`+0.091897`); its largest macro DA loss was D3 (`-0.034529`).

These are directional-accuracy comparisons only. They do not establish economic value or profitability.

## 13. LSTM seed-stability analysis

Population standard deviation (`ddof=0`) and range were computed across seeds for every fold/asset and for MAE, MAE Skill, and DA.

- Largest MAE seed dispersion: **D1/MSFT**, standard deviation `0.00206948`, range `0.00480437`.
- Smallest MAE seed dispersion: **D5/GOOGL**, standard deviation `0.00004968`, range `0.00012112`.
- MAE-Skill sign changed across seeds in **8/20** cells: D1 GOOGL, D1 NVDA, D2 AAPL, D4 GOOGL, D4 MSFT, D5 AAPL, D5 GOOGL, and D5 NVDA.

The largest dispersion is material relative to the observed skill margins, and eight sign changes show that conclusions in those cells depend on initialization. This is why the audit reports all seeds and the seed-mean metric rather than a best seed.

Full stability artifact: `results/combined/lstm_seed_stability_f965911f3b488f43e12e34996cef785e40de4d2d54fa52485f3e2f4064197d95.csv`.

## 14. Complexity comparison

| Model | Candidate fits | Refits | Total fits | Seeds | Determinism/variability | Observed development benefit |
|---|---:|---:|---:|---:|---|---|
| Baseline | 0 | 0 | 0 | n/a | Deterministic rules; no initialization | Reference comparator |
| LightGBM | 80 | 20 | 100 | 1 | Deterministic reference execution | Positive MAE Skill in 12/20 cells and 3/5 macro folds; inconsistent |
| LSTM | 120 | 60 | 180 | 3 | Deterministic execution attempted; initialization variability retained | Positive seed-mean MAE Skill in 4/20 cells and 0/5 macro folds |

Reference runtime was approximately 15.55 seconds for the full LightGBM runner and 1,064.876 seconds for the full LSTM runner. This is a factual comparison, not a formal cost-benefit score.

## 15. Supported development conclusions

The following conclusions are supported by the frozen D1–D5 evidence:

- The three model families were evaluated on exactly the same 5,032 development observations per seed where applicable.
- LightGBM produced small MAE gains in three macro folds but did not consistently beat zero-return persistence.
- LightGBM failed to beat zero-return persistence on macro MAE in D2 and D3; D3 also lost to direction persistence.
- The LSTM seed-mean failed to beat zero-return persistence on macro MAE in every development fold.
- LSTM regression performance was worse than persistence in 16/20 asset/fold cells, while directional gains occurred in 11/20, so MAE and DA tell different stories.
- D1/MSFT was the most initialization-sensitive LSTM cell by MAE dispersion; eight cells changed MAE-Skill sign across seeds.
- Model ranking varied by fold. Complexity did not yield a stable monotonic improvement over the baseline.

These are performance-defined failure cases. No regime explanation is attached to them.

## 16. Conclusions NOT yet supported

Phase 2G does **not** support any claim that:

- volatility, trend, a bear/bull market, or a regime transition caused a model failure;
- any model is universally superior or inferior outside these D1–D5 development folds;
- any metric difference is statistically significant under a valid dependence-aware test;
- any model is profitable after costs, tradable, calibrated for risk, or economically useful;
- PatchTST will outperform the existing families;
- the D1–D5 result generalizes to F1/2025;
- the optional family gate can be used to retune or replace frozen LightGBM/LSTM results.

## 17. PatchTST gate criteria G1–G7

The following criteria were fixed before recording the decision:

| Gate | Criterion |
|---|---|
| G1 | Baseline, LightGBM, and LSTM pipelines are complete, reproducible, and pass integrity checks. |
| G2 | The current benchmark does not already establish that added deep-model complexity is clearly unnecessary; a challenger must answer a distinct architecture question, not search for a winner. |
| G3 | PatchTST has a materially different time-series inductive bias from tree-based LightGBM and recurrent LSTM. |
| G4 | Available history, feature dimension, sequence availability, and asset count support a bounded experiment without synthetic data. |
| G5 | A tightly bounded, reproducible CPU-compatible pilot is practical. |
| G6 | A very small candidate space can be frozen without changing features, folds, target, assets, or tuning against test results. |
| G7 | Development can remain entirely within D1–D5 without touching F1/2025. |

No hidden criterion was used.

## 18. Result for each gate

| Gate | Result | Evidence |
|---|---|---|
| G1 | **PASS** | All three frozen families passed file/tree hashes, scope checks, exact-key checks, independent recomputation, and the test suite. |
| G2 | **PASS** | LightGBM gains were small/inconsistent; LSTM lacked robust MAE improvement. One corrected recurrent model does not exhaust the distinct patch-based time-series question. |
| G3 | **PASS** | The version 1.1 specification defines PatchTST as a patch-based transformer using longer temporal context, distinct from tabular trees and recurrence. |
| G4 | **PASS** | D1 contains 942 causal training feature rows per asset and 880 valid 63-session endpoints per asset—3,520 across four assets—with 17 frozen features, enough for a bounded pilot. |
| G5 | **PASS** | The larger three-seed LSTM benchmark completed locally on CPU in 1,064.876 seconds; a one-fold pilot with hard wall-time/memory limits is practical. This is not a measured PatchTST runtime. |
| G6 | **PASS** | The immutable feature, target, fold, schema, and selection contracts can be reused; a small architecture/context/seed/budget set can be frozen before fitting. |
| G7 | **PASS** | All present evidence is limited to 2020–2024 D1–D5, and a pilot can use a preregistered development fold only. |

## 19. Final PatchTST gate decision

**ADMIT PATCHTST TO A PREREGISTERED PILOT.**

The decision is frozen in `results/combined/patchtst_gate_decision_b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7.json`.

PatchTST remains unimplemented. Admission does not automatically authorize full D1–D5 execution. A separate pilot gate must be passed first.

## 20. Rationale

PatchTST is admitted for **methodological diversity**, not because the existing models performed poorly and not because superiority is expected. LightGBM tests nonlinear tabular feature interactions; LSTM tests a corrected recurrent sequence model; PatchTST would test whether patch-based temporal encoding and a different long-context inductive bias yield stable validation value under the same causal contract.

The development record leaves that question scientifically open while also warning against broad search: LightGBM gains are small and fold-dependent, the LSTM adds compute without robust MAE gains, and daily data are limited for a transformer. The appropriate response is therefore a frozen, bounded pilot with a separate stop/go criterion—not an unrestricted benchmark.

## 21. F1/2025 protection confirmation

**PASS.** No baseline, LightGBM, LSTM, combined, or PatchTST 2025 metric was computed. No F1 prediction was generated or inspected. Combined artifacts contain only D1–D5 and target years 2020–2024; no placeholder 2025 comparison row exists.

Mechanically retained 2025 rows in the frozen source datasets were not used for Phase 2G performance analysis.

## 22. Regime-analysis prohibition confirmation

**PASS.** No regime was labeled, joined, summarized, or used to explain performance.

- Baseline regime columns remain exactly `not_labeled_phase2c`.
- LightGBM and LSTM regime columns remain exactly `not_labeled_pre_regime_analysis`.
- No frozen prediction column was modified.
- Combined tables do not reinterpret sentinel values as categories.

Any later regime claim requires a separately frozen causal labeling and stress-test design.

## 23. Test-suite status

Entry gate before Phase 2G changes:

- total: 111
- passed: 111
- failed: 0
- skipped: 0
- warnings: 0

Final complete suite after adding 13 audit-only tests:

- total: 124
- passed: 124
- failed: 0
- skipped: 0
- warnings: 0
- runtime: 15.13 seconds

The new tests cover exact baseline/LightGBM and per-seed LSTM keys, three-seed enforcement, metric-level seed aggregation, equal-asset macro aggregation, arithmetic macro skill, win/loss tolerance, regime sentinel protection, 2025 exclusion, gate schema/enums, and a static prohibition on model/training imports and fit calls in Phase 2G.

## 24. Artifact locations/hashes

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `results/combined/development_asset_fold_comparison_61ad31f5e745338ccf679be9614188bab869964b350ab4f32a26067f85a2f3a1.csv` | 20 | `61ad31f5e745338ccf679be9614188bab869964b350ab4f32a26067f85a2f3a1` |
| `results/combined/development_macro_fold_comparison_042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6.csv` | 5 | `042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6` |
| `results/combined/cross_fold_summary_5d6eeac0aad1b87b735714f0238f711b97fa6abce5b77608e42b0c905d3daad8.csv` | 8 | `5d6eeac0aad1b87b735714f0238f711b97fa6abce5b77608e42b0c905d3daad8` |
| `results/combined/baseline_win_loss_counts_0e5210dfbf806fec4a065e138e115524c317242ed3a7c6e1b2e8068b3e858684.csv` | 10 | `0e5210dfbf806fec4a065e138e115524c317242ed3a7c6e1b2e8068b3e858684` |
| `results/combined/lstm_seed_stability_f965911f3b488f43e12e34996cef785e40de4d2d54fa52485f3e2f4064197d95.csv` | 20 | `f965911f3b488f43e12e34996cef785e40de4d2d54fa52485f3e2f4064197d95` |
| `results/combined/model_complexity_summary_e5d4ce6f88a686663da21b188f86bae92614075580ac7722486ab0844f8c05d6.csv` | 3 | `e5d4ce6f88a686663da21b188f86bae92614075580ac7722486ab0844f8c05d6` |
| `results/combined/development_fold_ranking_e13f0bd43d0b2f3ae80e41f8d0278e808a1cf04509b1eda8c1986e13d6826de9.csv` | 15 | `e13f0bd43d0b2f3ae80e41f8d0278e808a1cf04509b1eda8c1986e13d6826de9` |
| `results/combined/patchtst_gate_decision_b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7.json` | — | `b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7` |
| `results/combined/combined_verification_manifest_393be15393847bdcc48d9482a0d72f32437f1024e85523088dbc3194c4eabb02.json` | — | `393be15393847bdcc48d9482a0d72f32437f1024e85523088dbc3194c4eabb02` |

The nine-file combined tree has aggregate identity `c0dc9928c7f9381bf0980d6ccd4552d1bd053d41456a19b1c992fb66a2d63be2`. Every CSV row and both JSON artifacts reference the baseline, LightGBM, and LSTM prediction hashes; spec version/hash; preregistration identity; generation timestamp; and source-tree/revision identity.

New implementation/test files:

| File | Responsibility | SHA-256 |
|---|---|---|
| `src/evaluation/combined_audit.py` | Hash gates, exact-key checks, independent metrics, comparison tables, gate, post-write verification | `a13ec30f692f2c075a72219cabc5a5d20041cc0a4a681319a3ffcbc615a17e12` |
| `src/evaluation/run_combined_audit.py` | One-command audit-only runner | `c37040b141556496acf179dc3b66be3d300ce1959c810384c69b0ca59eb1cd9f` |
| `tests/test_combined_audit_phase2g.py` | Thirteen Phase 2G training-free tests | `c59d34d3b0369709c00de9698a68ddbe0d0b5635e0adfac2f3db7ca4ba79cdf6` |

The source-tree identity recorded at generation was `fe37172dca930c566701f5a8508952705ecb7895784baaae60fabeda519920e3` across 26 Python source files. Git revision metadata was unavailable because this repository snapshot has no usable `.git` metadata; the manifest records that limitation explicitly.

## 25. Deviations from specification

**None.** Phase 2G used the frozen version 1.1 target, features, folds, assets, metrics, candidate outcomes, and seeds. It did not modify or rerun model selection, average LSTM forecasts, inspect F1, or perform regime analysis.

The PatchTST decision follows the explicit G1–G7 Phase 2G gate and remains subordinate to the existing specification requirement for preregistration and a bounded D1 pilot before any full execution.

## 26. Remaining risks

- The PatchTST compute assessment is a feasibility inference from the completed local CPU benchmark, not a measured PatchTST pilot runtime or memory profile.
- D1 supplies limited daily observations for a transformer; a pilot may be unstable or fail its validation-value gate.
- Using D1–D5 evidence to admit an optional family makes any later PatchTST result conditional exploratory development evidence, not a replacement for preregistered core models.
- Five folds are too few for strong inferential claims, and daily observations are serially dependent.
- Directional accuracy does not encode return magnitude, turnover, costs, or risk.
- Regime definitions remain intentionally absent, so no robustness-under-regime-shift conclusion is available yet.
- Git commit identity is unavailable in this repository snapshot; reproducibility relies on content hashes and manifests.
- F1 remains protected, so final generalization is unknown by design.

## 27. Recommended next phase

Proceed next—only after explicit authorization—to:

> **Phase 2H — PatchTST Preregistration and Bounded Pilot**

The first Phase 2H action should be to create and freeze a PatchTST preregistration **before any fit**. It must freeze the architecture, context lengths, seeds, training budget, pilot fold, CPU/memory/time limits, preprocessing reuse, candidate-selection rule, and pilot acceptance criteria. The pilot must remain within development data and must not automatically lead to full D1–D5 execution; a separate pilot pass/stop decision is required.

Do not begin PatchTST implementation, regime analysis, or F1/2025 work automatically.
