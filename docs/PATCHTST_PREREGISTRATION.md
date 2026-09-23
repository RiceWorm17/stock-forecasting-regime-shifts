# PatchTST Preregistration

Status:  
**FROZEN BEFORE FIRST PATCHTST FIT**

Purpose:  
Define the optional PatchTST challenger and bounded pilot before any implementation or training.

Scope:  
The preregistration authorizes only a later bounded pilot.  
It does **NOT** authorize the full D1–D5 PatchTST benchmark.

## 1. Authorization and provenance

This Phase 2H-A document operationalizes, but does not alter, experiment specification version 1.1. The earlier learned-model preregistration deliberately left PatchTST closed until the required learned benchmarks and a later gate were complete. Phase 2G then evaluated G1–G7 as PASS and froze the decision **ADMIT PATCHTST TO A PREREGISTERED PILOT**.

That decision is bound to:

- `results/combined/patchtst_gate_decision_b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7.json`, SHA-256 `b46fb9ca62675417594a69fab761792370a867fbb9ba8979eb8928e45230cfb7`;
- `docs/audit/PHASE_2G_REPORT.md`, SHA-256 `e745662a3a8a8a8497348e2480e821a44ab4b55d7ff16721e3f42067db607667`;
- raw data SHA-256 `55f74f058493102387f1c2d0a848bd673a3cfa0473d617dfdb34aea8a94f7736`; and
- processed data SHA-256 `9352da6b434011f9844ef0acc1813ea33dce39152712752bba21a3005aabf7be`.

Phase 2H-A authorizes documentation and configuration only. It authorizes no implementation, dependency installation, model fit, pilot run, test prediction, regime analysis, or F1/2025 access. Phase 2H-B requires separate authorization.

## 2. Scientific role

PatchTST is included for **methodological diversity**. It is not included because LightGBM or LSTM performed poorly, because the project needs a model that beats the baseline, or because a Transformer is assumed to be superior.

The scientific question is:

> Does a patch-based, channel-independent temporal encoder provide meaningfully different forecasting behavior from zero-return persistence, tabular LightGBM, and recurrent LSTM under the same causal feature, target, fold, and evaluation contract?

No expected-superiority claim is preregistered. Pilot predictive performance cannot decide whether the technical pilot passes.

## 3. Prediction task and asset policy

The primary target is `target_log_return`: the next observed trading session's asset-local log return. The forecast horizon is one observed session. The only directional prediction is sign-derived:

`predicted_direction = 1 if predicted_log_return > 0 else 0`

There is no separate direction classifier and no target scaling.

One separate model is fitted for each of AAPL, MSFT, GOOGL, and NVDA. SPY remains a same-date market feature/reference only and is never a prediction target. Pooled multi-stock training is prohibited for the core challenger.

## 4. Frozen feature set

The input order is exactly:

1. `asset_log_return_1d`
2. `asset_log_return_lag_1`
3. `asset_return_mean_5`
4. `asset_return_std_5`
5. `asset_volume_mean_5`
6. `asset_volume_std_5`
7. `asset_return_mean_21`
8. `asset_return_std_21`
9. `asset_volume_mean_21`
10. `asset_volume_std_21`
11. `spy_log_return_1d`
12. `spy_log_return_lag_1`
13. `spy_return_mean_5`
14. `spy_volatility_5`
15. `spy_return_mean_21`
16. `spy_volatility_21`
17. `spy_trend_63`

No additional indicator or external feature is allowed. This explicitly excludes RSI, MACD, Bollinger Bands, fundamentals, news, sentiment, and macro data. Regime labels are not inputs. The feature set cannot change after pilot results are observed.

## 5. Exact model definition

The documentation name is **PatchTST-style channel-independent patch Transformer**. This is a small custom PyTorch model using the already-approved PyTorch ecosystem, not a claim of byte-for-byte reproduction of any external library.

The input tensor has shape `(batch, context_length, 17)`. The exact conceptual flow is:

1. transpose time/features into 17 feature-channel sequences;
2. extract right-aligned temporal patches independently within each channel;
3. apply one shared biased linear projection from 16 values to `d_model=32`;
4. add one shared learned positional embedding over chronological patch positions, zero-initialized;
5. apply the same two-layer Transformer encoder independently to every channel by treating batch × channel as the encoder batch;
6. restore `(batch, channel, patch, embedding)` axes;
7. flatten in channel, chronological-patch, embedding-dimension order;
8. apply head dropout 0.10; and
9. apply one linear scalar regression head.

There is no pooling search, attention-pooling search, alternate head, RevIN, or architecture grid.

## 6. Channel-independent design

Every feature channel uses the **same** patch projection, positional embedding, and Transformer encoder weights. The Transformer never receives all 17 channels flattened into a joint token sequence before channel-independent encoding. Cross-channel information is combined only by the final scalar linear head after the independently encoded channel/patch representations have been restored and flattened.

## 7. Right-aligned patch policy

For context length (L), patch length (P=16), and stride (S=8):

1. the final patch starts at `L - P` and ends at `L - 1`;
2. earlier starts are obtained by repeatedly subtracting `S` while the start remains nonnegative; and
3. the resulting starts are reversed into chronological order.

Thus the number of patches is `1 + floor((L - P) / S)`. No padding is permitted. The last patch always contains the origin-date timestep. Early context rows not covered by this right-aligned grid remain unused. A naive left-aligned `unfold()` that omits the origin-date tail is prohibited.

Frozen grids:

| Candidate | L | Chronological starts | Patches | Final end | Unused prefix |
|---|---:|---|---:|---:|---:|
| PATCHTST_63 | 63 | 7, 15, 23, 31, 39, 47 | 6 | 62 | 7 timesteps |
| PATCHTST_126 | 126 | 6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110 | 14 | 125 | 6 timesteps |

## 8. Frozen architecture

| Component | Frozen value |
|---|---|
| Input channels | 17 |
| Patch length / stride | 16 / 8 |
| Patch projection | shared `Linear(16, 32, bias=True)` |
| Positional information | shared, learned, zero-initialized `(1, n_patches, 32)` parameter |
| `d_model` | 32 |
| Attention heads | 4 |
| Encoder layers | 2 |
| Feed-forward dimension | 64 |
| Transformer dropout | 0.10 |
| Activation | GELU |
| Encoder layout | `batch_first=True`, `norm_first=True` |
| Layer-norm epsilon | `1e-5` |
| Final encoder layer norm | yes |
| Head | flatten, dropout 0.10, scalar linear output |
| Head input, context 63 | `17 × 6 × 32 = 3,264` |
| Head input, context 126 | `17 × 14 × 32 = 7,616` |
| RevIN | none |
| Output / dtype | one float32 value per sample |

All architecture fields are fixed; none is a pilot search dimension.

## 9. Context candidates

Exactly two candidates are allowed:

- `PATCHTST_63`: context length 63, six patches;
- `PATCHTST_126`: context length 126, fourteen patches.

Both use the identical architecture, optimizer, preprocessing, and training contract. No context, patch length, stride, model dimension, head count, layer count, feed-forward dimension, or head alternative may be added after results are observed.

## 10. Preprocessing contract

Only the 17 input features are standardized; targets are never standardized. For each fold × asset, the candidate-stage `StandardScaler` is fitted once on training feature rows only and shared by both contexts and every seed.

Each canonical feature row contributes once to the mean and population variance (`ddof=0`), regardless of how often it later appears in overlapping sequences or patches. Validation and test statistics cannot enter scaling. A zero-variance feature is retained and centered with effective scale 1.0.

If a full benchmark is later authorized, the winning-refit scaler is newly fitted on train-plus-validation feature rows only. Test statistics remain prohibited.

## 11. Sequence and timing contract

Sequences are asset-local, chronological, explicit-key based, unpadded, and end at `origin_date`. Every element must be one of the 17 finite feature values observable at or before that origin. Targets and future features cannot be inputs, and no sequence may cross an asset boundary.

Validation context may include earlier training feature rows. Validation targets are excluded from training, preprocessing, and model updates. For a given asset, both candidates and both pilot seeds must produce exactly the same D1 validation endpoint keys. Intersection-based shrinking is prohibited.

## 12. Training contract

| Field | Frozen value |
|---|---|
| Loss | `torch.nn.L1Loss(reduction="mean")` |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Betas | (0.9, 0.999) |
| Epsilon | `1e-8` |
| Weight decay | 0.0001 |
| AMSGrad | false |
| Batch size | 64 |
| Maximum epochs | 40 |
| Shuffle / drop last | false / false |
| Data-loader workers | 0 |
| Gradient clipping | maximum norm 1.0 |
| Scheduler / warm-up | none / none |

Early stopping monitors full validation MAE calculated over individual rows, not an unweighted mean of batch losses. Improvement means `validation_mae < best_mae - 0.00001`. Training stops after six consecutive completed epochs without improvement or after epoch 40, restores the best state, and records a positive one-based best epoch. Nonfinite loss, prediction, or metric fails P3. There is no learning-rate search.

## 13. Determinism and reference environment

The reference pilot is CPU-only. Python, NumPy, and PyTorch CPU are seeded. PyTorch deterministic behavior is requested where supported. Intra-op and inter-op thread counts are both one; the inter-op value is set once at process start. Hardware and thread policy cannot change based on performance, and there is no GPU rescue if the frozen CPU gate fails.

The existing, unmodified reference environment observed during Phase 2H-A is Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, and PyYAML 6.0.3, with CUDA unavailable. A content-addressed environment manifest must be captured and verified before the first future fit.

## 14. Pilot fold and seeds

The bounded pilot uses D1 only:

- training targets: 2015-01-01 through 2018-12-31;
- validation targets: 2019-01-01 through 2019-12-31;
- pilot seeds: 1729 and 2718.

D1 test/2020, every D2–D5 validation/test result, and F1/2025 are prohibited. No development-test prediction, refit, or winner test evaluation is allowed. Full-benchmark seed 31415 is intentionally absent from the pilot; pilot results cannot remove or replace it.

## 15. Pilot fit accounting

The complete candidate grid is:

`2 contexts × 2 pilot seeds × 4 assets = 16 candidate fits`

All 16 must complete. There are no refits and no test predictions.

P5 additionally requires one fresh deterministic replay of `PATCHTST_63`, AAPL, seed 1729. This replay is a verification execution, not a seventeenth candidate-grid member and produces no candidate-selection evidence. It is outside the frozen **16 candidate-fit** budget but inside the 1,800-second end-to-end pilot wall-time measurement. Therefore the future command has 16 candidate-grid fits and 17 total training executions including the mandatory P5 replay. This explicit distinction reconciles the complete-grid and reproducibility requirements without expanding the search.

## 16. Pilot outputs

For every candidate × asset × pilot-seed slice, record validation MAE, RMSE, sign-derived DA, positive class balance, best epoch, epochs run, runtime, and all warning/error status and messages. Preserve row-level validation predictions with explicit keys so P4 and P5 can be checked.

These values provide technical visibility only. They cannot alter the architecture, contexts, patching, optimizer, learning rate, features, seed sets, or resource limits.

## 17. Frozen P1–P7 acceptance gate

The pilot passes only when every criterion is PASS:

| Gate | Criterion | Frozen PASS rule |
|---|---|---|
| P1 | Integrity | Frozen data, historical configs/preregistration, Phase 2G gate, Phase 2H-A documents, and recorded hashes are unchanged. |
| P2 | Complete grid | All 16/16 candidate-grid fits complete across two contexts, two seeds, and four assets. |
| P3 | Numerical validity | All losses, predictions, and required metrics are finite; no reproducible training failure remains. |
| P4 | Exact validation coverage | Both contexts and seeds have identical canonical D1 validation keys per asset, with no intersection/shrinkage. |
| P5 | Deterministic reproducibility | A fresh PATCHTST_63/AAPL/1729 replay reproduces validation keys, best epoch, and epochs run exactly, and predictions plus MAE/RMSE/DA with `rtol=1e-7`, `atol=1e-9`. |
| P6 | Resource feasibility | Entire pilot command, including grid, replay, verification, writes, and overhead, completes in at most 1,800 seconds on the frozen CPU policy. Peak process memory is recorded where practical; measurement unavailability is disclosed but is not by itself a failure. |
| P7 | Full-benchmark feasibility estimate | `(pilot_total_seconds / 16) × 180 × 1.25 ≤ 21,600 seconds`. The 180 comprises 120 candidate fits plus 60 refits. |

The P7 divisor remains the 16-member candidate grid as preregistered. Because P6's measured total includes the extra reproducibility replay and overhead, the resulting extrapolation is conservative.

## 18. No performance-based pilot gate

The gate contains **no predictive-performance threshold**. PatchTST need not beat zero-return persistence, LightGBM, or LSTM; need not have positive MAE Skill; and need not reach any DA threshold. Weak validation performance is allowed.

Using favorable pilot performance as a go/no-go rule would turn a technical feasibility pilot into model shopping. P1–P7 therefore assess only integrity, completeness, numerical correctness, exact coverage, reproducibility, and resource feasibility.

## 19. Pilot failure policy

If any P1–P7 criterion fails, record **PILOT NOT PASSED** and do not admit PatchTST to the full D1–D5 benchmark. Do not shrink the model, remove a context/asset/seed, increase a time cap, change the optimizer or architecture, add GPU rescue, or relax the resource gate after failure. Continue the study without PatchTST. A failed pilot is not a project failure.

## 20. Full-benchmark seeds and budget

If and only if the pilot passes and a later full benchmark is separately authorized, the seeds are exactly 1729, 2718, and 31415. No seed may be added, removed, selected as best, or ensembled in the core benchmark.

The maximum full grid is `2 contexts × 3 seeds × 5 folds × 4 assets = 120 candidate fits`, followed by exactly `5 × 4 × 3 = 60` winning-context refits. The maximum is 180 total fits. There is no grid expansion or extra architecture experiment.

## 21. Future full-benchmark selection rule

Within each fold and for each context candidate:

1. compute validation MAE separately for each seed and asset;
2. take the arithmetic mean over the three seeds within each asset; and
3. take the equal-weight arithmetic mean over AAPL, MSFT, GOOGL, and NVDA.

The lowest macro validation MAE wins. Values within 0.00001 are a practical tie, resolved in this order:

1. shorter context (`PATCHTST_63` before `PATCHTST_126`);
2. lower macro validation RMSE calculated with the same seed-then-asset aggregation; and
3. ascending lexical `config_id`.

Development-test metrics, F1 results, and regime performance are prohibited selection inputs. Passing the pilot does not itself authorize this full run.

## 22. Future full-benchmark refit rule

For each fold × asset × seed, retrieve that exact seed/asset's best epoch from the winning context candidate. Initialize a fresh model, fit a fresh scaler on train-plus-validation features only, and train on train plus validation for exactly the recorded epoch count. There is no test early stopping, test evaluation set, test-target feedback, or within-test-year parameter update.

## 23. Future prediction contract

If separately authorized, PatchTST development-test rows use the existing 18 fields in this exact order:

1. `run_id`
2. `spec_version`
3. `data_version`
4. `model`
5. `model_config_id`
6. `seed`
7. `fold`
8. `partition`
9. `asset`
10. `origin_date`
11. `target_date`
12. `actual_log_return`
13. `predicted_log_return`
14. `actual_direction`
15. `predicted_direction`
16. `trend_regime`
17. `volatility_regime`
18. `transition_regime`

`model=patchtst`, `partition=test`, and every regime field remains `not_labeled_pre_regime_analysis`. A future full D1–D5 artifact must match all 5,032 canonical development keys separately for each seed. F1/2025 remains prohibited.

## 24. Resource limits

The pilot candidate-fit budget is 16, maximum epochs are 40, and total pilot wall time is capped at 1,800 seconds. The mandatory P5 replay is not another candidate but is included in measured pilot wall time. The full-runtime estimate uses a 1.25 safety factor and must not exceed 21,600 seconds:

`estimated_full_seconds = (pilot_total_seconds / 16) × 180 × 1.25`

No GPU rescue, grid expansion, time-cap increase, or extra architecture experiment is permitted after pilot execution begins.

## 25. Dependency policy

Phase 2H-A installs nothing. The later implementation must first use the existing PyTorch CPU ecosystem and a small custom implementation. Do not add `transformers`, `neuralforecast`, `tsai`, `pytorch-forecasting`, or TensorFlow.

Only if custom implementation is proven impossible **before any pilot fit** may a reviewed preregistration amendment propose an external framework. No dependency change may follow inspection of pilot performance.

## 26. F1 and regime protection

The pilot cannot access D1 test/2020, D2–D5 development results, or F1/2025. It cannot generate development-test forecasts. F1 remains locked until the separate final-test procedure.

Regime remains a later evaluation dimension. It cannot affect features, fitting, scaling, early stopping, candidate selection, the pilot gate, or explanations in this phase. The frozen learned-model sentinel remains `not_labeled_pre_regime_analysis` until a separately preregistered causal join creates a new artifact.

## 27. Change control and next gate

The architecture, contexts, seeds, preprocessing, training protocol, D1 pilot boundary, resource limits, P1–P7 rules, future selection procedure, and future refit procedure are frozen before the first fit. A necessary correction before fitting requires an explicit reviewed amendment; performance-motivated change after fitting is prohibited.

The next possible phase is **Phase 2H-B / PatchTST Bounded Pilot Implementation and Execution**, but it does not begin automatically. Phase 2H-B may implement and execute only the bounded pilot defined here. Even a passed pilot requires a separate review and authorization before any full D1–D5 PatchTST benchmark.
