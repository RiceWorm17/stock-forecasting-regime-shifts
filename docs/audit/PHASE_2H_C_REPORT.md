# Phase 2H-C — PatchTST Full-Benchmark Authorization Review

## 1. Phase status

**PHASE 2H-C COMPLETE — FULL BENCHMARK DECLINED.** The final decision is `DECLINE_FULL_PATCHTST_BENCHMARK` because A2 and A7 fail. The frozen all-pass rule permits authorization only when A1 through A8 all pass. This review does not revoke the narrower Phase 2H-B finding that the saved pilot evidence passes P1 through P7.

## 2. Review-only scope

This phase performed evidence review, source inspection, saved-artifact verification, checkpoint-structure inspection, correctness-only output diagnostics, tests, and hashing. It performed **zero model fits**. It did not rerun the pilot, execute the full PatchTST benchmark, refit a historical model, generate development-test predictions, evaluate F1/2025 performance, perform regime analysis, or use a predictive-performance threshold. No existing source, configuration, preregistration, historical-result, combined-result, or pilot file was modified. The only new artifacts are this report and the authorization JSON identified in Section 31.

## 3. Frozen identities

All authoritative identities passed both the entry check and the final post-artifact check.

| Frozen item | Files | SHA-256 |
|---|---:|---|
| Baseline tree, `results/baselines` | 6 | `1f42bb6758b54251060983e1b7a9323861d94ac768ee5371571e77e5b46fc466` |
| LightGBM tree, `results/lightgbm` | 113 | `6f1079692067a8ed49dcce3d3db52efd22e00883084575a3a804a6f942e9d970` |
| LSTM tree, `results/lstm` | 235 | `64d0866469259062e49652c17be364e34cde6bbe7fecc17dbb821489e6b66f61` |
| Phase 2G combined tree, `results/combined` | 9 | `c0dc9928c7f9381bf0980d6ccd4552d1bd053d41456a19b1c992fb66a2d63be2` |
| Phase 2G report | 1 | `e745662a3a8a8a8497348e2480e821a44ab4b55d7ff16721e3f42067db607667` |
| Phase 2G PatchTST gate | 1 | `b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7` |
| Phase 2H-A report | 1 | `72d80bdc0725682754a3d939070865d7d96564a290919b8f1a6bf05442321f95` |
| PatchTST preregistration | 1 | `adb4be727379d005dc69fd6d8528c6e46abda993394f21a946097e59cabfb486` |
| PatchTST search configuration | 1 | `6be2b9e2017c1f7b89c553ef1f65eaa995e6cd2cbc233fa2cd84a51a12c91124` |
| Phase 2H-B report | 1 | `7f510968c7a019476ff59f2629fec5b1a02478e18b6938cd2d36d8e96d03c50f` |
| Phase 2H-B run manifest | 1 | `f15fcaf1e32a018f297caea16a930a429cfe7b7f4ab205f75b6e6e94b4b8bafe` |
| Phase 2H-B pilot gate | 1 | `7b1da95b984d0ebe3b50d15fcf48f17b536438c957dc2071162f541c57d50601` |
| Phase 2H-B pilot tree, `results/patchtst/pilot` | 34 | `97b630d8d9b9ab889c020152eb29dc1555fe7a33a42c99f30d1aabe0371941ce` |
| Python source tree | 29 | `c7ed09a3eaa487be2c5039352c348d522650a50be4063e9f36b256b2e1c3ee25` |

The content hashes of the pilot run manifest and gate also match the hash embedded in their filenames. The old pilot gate still records `full_benchmark_authorized: false` and was not rewritten.

## 4. Independent P1–P7 verification

The review reloaded saved evidence rather than accepting report prose. All 33 artifact references in the run manifest were found and hash-verified.

| Gate | Result | Independent evidence |
|---|---|---|
| P1 — Integrity | **PASS** | Frozen inputs, artifact hashes, source identity, row keys, dates, and canonical actual values are internally consistent. |
| P2 — Complete grid | **PASS** | Exactly 16 candidate identities are present: 2 contexts × 4 assets × 2 pilot seeds. The evidence has 16 metric rows and 16 checkpoints. |
| P3 — Numerical validity | **PASS** | All 291 epoch records and all saved losses, gradients, parameters, predictions, and required metrics are finite and consistent. |
| P4 — D1 validation coverage | **PASS** | Every candidate/asset/seed slice has the exact 252 canonical rows. The 16 slices contain 4,032 rows; the largest independently recomputed metric difference is `1.1102230246251565e-16`. |
| P5 — Deterministic replay | **PASS** | The fresh saved `PATCHTST_63`/AAPL/1729 replay matches keys, best epoch, epoch count, predictions, and metrics; maximum absolute prediction difference is `0.0`. |
| P6 — Pilot feasibility | **PASS** | `788.0661089000059` seconds is below the `1,800`-second cap. |
| P7 — Full estimate | **PASS** | The independent calculation equals the saved estimate exactly: `11,082.179656406333` seconds, below the `21,600`-second cap. |

The validation keys are ordered, all origin dates precede target dates, every actual value matches the canonical processed target to serialization precision, and every saved direction equals the strict sign of the corresponding actual or prediction.

## 5. Pilot implementation-conformance audit

The frozen PatchTST model, feature, target, patching, architecture, training, scaling, and checkpoint contracts are implemented as specified. Checkpoint inspection found 16 finite state dictionaries, each with the expected 31 keys, projection shape `[32, 16]`, positional shape `[1, 6, 32]` or `[1, 14, 32]`, and output-head shape `[1, 3264]` or `[1, 7616]`. No hidden head or RevIN state exists.

Exact conformance nevertheless **fails** because `src/evaluation/run_patchtst_pilot.py:384-404` loads the entire processed CSV—with `target_log_return`, `target_direction`, and all 17 frozen features—before computing the date mask and retaining the authorized 2015–2019 rows. The read materialized 6,032 post-2019 rows before exclusion. Those rows did not reach scaling, sequence construction, fitting, validation, predictions, metrics, replay comparison, or model selection, so no scientific leakage into the saved pilot results was found. However, `docs/PATCHTST_PREREGISTRATION.md:302` and `configs/patchtst_search.yaml:344-349` literally prohibit D1-test/2020, D2–D5, and F1/2025 access. The materialization is therefore a genuine authorization blocker, not a wording preference. No silent correction was made.

## 6. Right-aligned patch audit

Patching conforms to the frozen rule: patch length `16`, stride `8`, no padding, and the final patch includes the origin timestep. For context `63`, starts are exactly `[7, 15, 23, 31, 39, 47]`, yielding 6 patches and a 3,264-value head input. For context `126`, starts are exactly `[6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110]`, yielding 14 patches and a 7,616-value head input. The implementation does not use a naive left-aligned unfolding that drops the origin-date tail.

## 7. Channel-independent encoder audit

The implementation uses one shared patch projection, one shared positional embedding, and one shared Transformer encoder. The 17 feature channels are folded into the encoder batch dimension, so no cross-channel attention occurs inside the encoder. Channel outputs are combined only when the encoded tensor is flattened into the final scalar regression head. This matches the preregistered channel-independent design.

## 8. Training-contract audit

The training path matches the frozen contract: L1 loss; AdamW; learning rate `0.001`; weight decay `0.0001`; batch size `64`; maximum 40 epochs; gradient clipping at `1.0`; `shuffle=false`; no scheduler; validation-row MAE monitoring; patience of 6 completed epochs; minimum improvement `0.00001`; and restoration of the best state. The architecture is fixed at `d_model=32`, 4 attention heads, 2 encoder layers, feed-forward width 64, dropout 0.10, GELU activation, `norm_first=true`, and final encoder normalization. It has neither RevIN nor a hidden head layer.

## 9. Scaling/target-unit audit

The 17 frozen input features are the only model inputs. Separate asset models are used. The asset scaler is fitted once per asset from authorized D1 training feature rows counted canonically once, then shared across both contexts and both pilot seeds. Validation rows do not enter the scaler. The target is the dimensionless fractional natural-log return `ln(close[target_date] / close[origin_date])`; it is never fitted to or transformed by a scaler. Training converts it to float32 only. L1 loss compares the model's direct scalar output against this unscaled target, and prediction saving performs no inverse transform or percentage conversion.

This scaling path is correct after the date filter. The separate ingestion violation in Section 5 remains: protected rows were read before being filtered, even though they were not used by the scaler or model.

## 10. Output-scale diagnostic review

**`OUTPUT_SCALE_CORRECTNESS = PASS_WITH_POOR_PREDICTIVE_BEHAVIOR`.** This is a correctness diagnosis, not a performance gate.

| Distribution | Rows | Mean | Population SD | Minimum | Maximum | Maximum absolute |
|---|---:|---:|---:|---:|---:|---:|
| Canonical D1 validation actuals, unique by key | 1,008 | `0.0018947` | `0.0180877` | `-0.1487844` | `0.0918517` | `0.1487844` |
| Saved candidate predictions | 4,032 | `0.0068417` | `0.4759357` | `-2.4049270` | `2.4946301` | `2.4946301` |

There are 242 predictions with absolute value above 1 and 861 above 0.5. The context-63 absolute-magnitude median/95th percentile are `0.1494990`/`0.6715138`; the context-126 values are `0.2743598`/`1.2822978`. The context dependence is inconsistent with a uniform ×100 conversion signature and is consistent with broad, uncalibrated outputs from the frozen head.

The six required correctness answers are:

1. **Yes.** PatchTST predicts the same canonical `target_log_return` unit used by the baseline, LightGBM, and LSTM pipelines.
2. **No.** No accidental multiplication, percentage conversion, exponential conversion, inverse transform, or feature-scaler misuse exists in the output path.
3. **No.** The target is not standardized during training and is not inverse-transformed when predictions are saved.
4. **Yes.** The unrestricted final `Linear(..., 1)` head directly represents an unscaled next-day log return.
5. **Yes.** Saved validation actuals equal canonical processed `target_log_return` values to a maximum absolute CSV round-trip difference of approximately `1.01e-16`, with exact canonical date keys and directions.
6. **Yes.** The extreme magnitudes are explainable as poor behavior of an unrestricted linear head over 3,264 or 7,616 flattened encoded values, not as a data-unit or alignment defect.

## 11. Memory-measurement limitation

Peak-process-memory measurement was attempted through `Windows_PSAPI_PeakWorkingSetSize` but returned no value because `GetProcessMemoryInfo` failed. No out-of-memory event, crash, or other resource failure was observed. The preregistered P6 rule explicitly states that unavailable reliable memory measurement alone does not fail the pilot, and the wall-time gate passed. No post-hoc memory threshold was invented. The missing measurement is accepted as a nonblocking limitation, although it reduces confidence in forecasting peak RAM for a full run.

## 12. Pilot wall-time verification

The authoritative total is `788.0661089000059` seconds. It comprises `783.0661089000059` seconds observed through training, replay, verification, and primary writes plus a conservative 5-second finalization allowance. It covers 16 candidate fits and one mandatory deterministic replay. The value is below the P6 cap by `1,011.9338910999941` seconds. P6 is **PASS**.

## 13. Full-runtime estimate verification

The frozen estimate was independently recomputed without model execution:

`(788.0661089000059 / 16) × 180 × 1.25 = 11082.179656406333 seconds`

This is approximately `3.078383237890648` hours and is `10,517.820343593667` seconds below the six-hour (`21,600`-second) cap. The saved and independently calculated values differ by `0.0` seconds. P7 is **PASS**. Resource feasibility does not override the conformance failures.

## 14. Full candidate-grid readiness

The future grid is sufficiently frozen: 2 contexts × 3 seeds × 5 folds × 4 assets = 120 candidate fits. It is followed by 5 folds × 4 assets × 3 seeds = 60 winning-context refits, for a maximum of 180 fitted models. The folds are D1–D5; candidates are `PATCHTST_63` and `PATCHTST_126`; assets are AAPL, MSFT, GOOGL, and NVDA. No grid expansion, extra context, architecture experiment, extra feature, asset substitution, or seed search is allowed.

Protocol readiness is **PASS**, but readiness is not execution authorization. The protected-ingestion defect must not be silently fixed and the benchmark must not run under this declined decision.

## 15. Full seed policy

The full seeds are exactly `1729`, `2718`, and `31415`. Seed 31415 was intentionally absent from the bounded pilot and remains mandatory for the full protocol. Seeds may not be added, removed, ranked, selected, or ensembled based on pilot or development performance. Candidate selection aggregates all three seeds according to Section 16; each winning refit retains its own seed and seed-specific epoch count.

## 16. Full selection rule

Selection is independent within each fold. For each candidate: (1) compute validation MAE separately for every seed and asset; (2) take the arithmetic mean over the three seeds within each asset; and (3) take an equal-weight arithmetic mean over the four assets. The lowest macro validation MAE wins.

An absolute macro-MAE difference of at most `0.00001` is a practical tie. Ties are resolved, in order, by: shorter context (`PATCHTST_63` before `PATCHTST_126`); lower macro validation RMSE using the same seed-then-asset aggregation; then ascending lexical `config_id`.

Test results, pilot D1-test information, later folds, directional accuracy, runtime, comparisons with LightGBM or LSTM, regime information, and F1/2025 are prohibited selection inputs. No pilot-performance comparison was used in this authorization decision.

## 17. Full refit rule

For each fold × asset × seed, the frozen rule takes `best_epoch` from that same seed and asset's validation fit for the winning context. It then fits a new scaler on train-plus-validation features only, initializes a fresh model with the same seed, and trains the train-plus-validation rows for exactly the recorded epoch count. It does not continue candidate weights. There is no test early stopping, test evaluation set, test-target access, or test-driven parameter update. Parameters remain fixed for the complete test year.

## 18. Canonical future artifact contract

Any future authorized development prediction artifact must use `model=patchtst`, `partition=test`, and contain these 18 fields: `run_id`, `spec_version`, `data_version`, `model`, `model_config_id`, `seed`, `fold`, `partition`, `asset`, `origin_date`, `target_date`, `actual_log_return`, `predicted_log_return`, `actual_direction`, `predicted_direction`, `trend_regime`, `volatility_regime`, and `transition_regime`. All regime fields remain `not_labeled_pre_regime_analysis`. Each seed must match all 5,032 canonical D1–D5 development keys. Required metrics and audit artifacts are already specified. F1/2025 is excluded.

## 19. F1 protection

F1/2025 performance was **not evaluated** in Phase 2H-C or by the saved pilot. No F1 prediction, metric, selection input, fit, or parameter update was produced. However, the pilot loader's full-file read materialized post-2019 feature and target columns—including F1 rows—before discarding them. Therefore the narrower statement “F1 performance was not evaluated or used” remains true, while the frozen literal statement “F1/2025 was not accessed” is not defensible. This distinction is the basis for A7 **FAIL**. This review did not evaluate F1 performance.

## 20. Regime protection

No regime analysis was performed. Regime labels were not used as features, training inputs, scaling inputs, routing rules, selection inputs, pilot gates, or authorization criteria. The future prediction-contract placeholders remain `not_labeled_pre_regime_analysis`. Regime stays reserved for a separately preregistered, post-prediction evaluation dimension.

## 21. Scientific role of PatchTST

PatchTST remains a methodological-diversity challenger. A patch-based, channel-independent Transformer supplies a temporal inductive bias distinct from a persistence baseline, tabular LightGBM, and recurrent LSTM. Its role is not justified by a desire to beat the baseline, compensate for another model, rescue portfolio results, or react to favorable or unfavorable pilot metrics. No predictive-performance comparison affected A1–A8.

## 22. A1 result

**A1 — Pilot evidence integrity: PASS.** P1 through P7 were independently reproduced from saved evidence, including artifact hashes, coverage, numerical checks, deterministic replay identity, elapsed time, and the full-runtime calculation.

## 23. A2 result

**A2 — Implementation conformance: FAIL.** The model, architecture, training, scaling, target, and patching implementation conform, but exact implementation conformance also includes the frozen access boundary. `load_d1_pilot_rows` reads the complete processed file before its date filter, contrary to the explicit no-access rule. A genuine mismatch is an authorization blocker even without metric contamination.

## 24. A3 result

**A3 — Output-scale correctness: PASS.** The detailed status is `PASS_WITH_POOR_PREDICTIVE_BEHAVIOR`. No target-unit, target-scaling, inverse-transform, date-alignment, endpoint-alignment, or head-output defect was found. Poor forecasts alone do not fail this criterion.

## 25. A4 result

**A4 — Resource feasibility: PASS.** P6 and P7 pass. The conservative 11,082.180-second estimate is approximately 3.08 hours and remains below the six-hour cap. The unavailable peak-memory reading is an accepted frozen-gate limitation, not a newly invented failure.

## 26. A5 result

**A5 — Full protocol completeness: PASS.** Candidate identities, folds, assets, seeds, fit counts, selection hierarchy, tie rule, refit procedure, metrics, prediction schema, artifact expectations, test-year freeze, and F1 prohibition were fixed before this review. No material scientific choice remains unspecified.

## 27. A6 result

**A6 — Scope integrity: PASS.** The planned scientific scope requires no new feature, architecture, target, fold, asset, seed, metric, selection rule, or refit rule. The discovered loader defect is a conformance problem, not permission to alter the scientific scope or pilot-established rules.

## 28. A7 result

**A7 — Protected-data integrity: FAIL.** No protected performance was evaluated or used, and no development-test prediction was generated. Nevertheless, the pilot process materialized 6,032 post-2019 rows—including protected feature and target columns—before exclusion. This violates the literal preregistered prohibition on D1-test/2020, D2–D5, and F1/2025 access.

## 29. A8 result

**A8 — Scientific-role integrity: PASS.** PatchTST's stated role remains methodological diversity. Neither its observed pilot behavior nor prior-model performance was used as an authorization threshold or rationale.

## 30. Final authorization decision

**`DECLINE_FULL_PATCHTST_BENCHMARK`.** A1=PASS, A2=FAIL, A3=PASS, A4=PASS, A5=PASS, A6=PASS, A7=FAIL, and A8=PASS. Because the frozen decision rule requires all eight criteria to pass, the already-preregistered full D1–D5 PatchTST benchmark is not authorized. This decision does not authorize any PatchTST training, F1/2025 access, regime analysis, architecture change, additional context, feature, seed, or corrective pilot.

## 31. Authorization artifact/hash

The new immutable decision artifact is:

`results/patchtst/authorization/patchtst_full_benchmark_authorization_5b63c323c9b28ddd1aa666365fb9df785fc21e6d45705317624e2ac77c811405.json`

Its SHA-256 is `5b63c323c9b28ddd1aa666365fb9df785fc21e6d45705317624e2ac77c811405`, exactly matching the filename. It records the frozen identities, independent P1–P7 results, output-scale diagnosis, resource and memory evidence, protocol readiness, A1–A8 decisions, specification deviation, source-tree identity, protected-data status, and final decline decision. The Phase 2H-B pilot gate remains unchanged and still says `full_benchmark_authorized: false`.

## 32. Test-suite result

The required entry run completed with 159 passed, 0 failed, 0 skipped, and 0 warnings (`159 passed in 30.30s`). After the authorization artifact was created, the required final run again completed with **159 passed, 0 failed, 0 skipped, and 0 warnings** (`159 passed in 22.06s`). No new test or review code was needed. The test runs did not execute a real pilot, candidate grid, benchmark, or refit; model fits executed during this review remain 0.

## 33. Historical immutability result

**PASS.** The final tree identities exactly equal the frozen values in Section 3: baseline 6 files, LightGBM 113, LSTM 235, Phase 2G combined 9, and Phase 2H-B pilot 34. The PatchTST preregistration, PatchTST search configuration, pilot run manifest, pilot gate, Phase 2G report, Phase 2H-A report, and Phase 2H-B report hashes also remain exact. The Python source tree remains 29 files with SHA-256 `c7ed09a3eaa487be2c5039352c348d522650a50be4063e9f36b256b2e1c3ee25`. Historical benchmark artifacts modified: **No**. Pilot artifacts modified: **No**.

## 34. Specification deviations

**Yes — one confirmed pre-existing implementation deviation was discovered.** `src/evaluation/run_patchtst_pilot.py:396` reads the full processed dataset before `src/evaluation/run_patchtst_pilot.py:397-400` computes and applies the 2015–2019 authorization mask. This contradicts the frozen no-access language. The effect is limited but material for governance: protected rows did not affect training, validation, inference, metrics, replay, or selection, yet exact implementation conformance and protected-data integrity fail.

No deviation was introduced or corrected in Phase 2H-C. A future reconsideration of PatchTST would require an explicit reviewed preregistration amendment and a fresh bounded pilot whose loader prevents protected rows from being materialized; this report does not authorize that work.

## 35. Remaining risks

The full PatchTST run remains unauthorized. The largest technical risk is repeating the protected full-file ingestion across future folds or accidentally converting protected materialization into scientific use. The saved pilot also shows very broad predictions, especially for the 126-day context; although this is correctness-valid poor behavior rather than a gate failure, it may produce unstable errors in later periods. Peak memory remains unmeasured, so the runtime estimate does not establish a RAM bound. Finally, any attempt to repair the loader after observing pilot outputs must be governed as a disclosed amendment with a new bounded pilot, not an invisible code edit or retroactive validation claim.

## 36. Recommended next phase

Proceed next with **`Phase 2I / Regime Label Preregistration and Development Stress-Test Design`**. Do not begin it automatically, and do not execute the declined PatchTST full benchmark. If PatchTST is reconsidered later, handle the protected-ingestion correction through a separate, explicit amendment-and-repilot decision path.
