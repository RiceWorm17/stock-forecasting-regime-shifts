Status: FROZEN BEFORE LEARNED-MODEL TRAINING
Experiment specification: 1.1
Raw dataset SHA-256: 55f74f058493102387f1c2d0a848bd673a3cfa0473d617dfdb34aea8a94f7736
Processed dataset SHA-256: 9352da6b434011f9844ef0acc1813ea33dce39152712752bba21a3005aabf7be

Material changes after learned-model results are observed are prohibited unless documented as a new exploratory specification rather than silently replacing the preregistered benchmark.

# Learned Model Preregistration

## 1. Authority, purpose, and scope

This document is the implementation contract for the required LightGBM and LSTM benchmarks in *When Stock Forecasting Models Fail: Stress-Testing Robustness Under Market Regime Shifts*. It narrows choices that [`EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md) version 1.1 requires to be fixed before learned-model development; it does not amend that frozen specification. If this document and version 1.1 ever appear to conflict, implementation must stop and the conflict must be resolved through documented change control before training.

This preregistration was frozen before either learned model was implemented, installed, trained, or evaluated. Phase 2D does not authorize learned-model predictions, F1/2025 evaluation, or regime-performance analysis.

The following are fixed for the core benchmark:

- prediction target and forecast timing;
- assets and separate-per-asset fitting policy;
- the 17 predictive features and their order;
- D1–D5 and F1 fold boundaries;
- family-specific candidate spaces;
- fold-local selection and tie-breaking rules;
- LightGBM early-stopping and refit rules;
- LSTM architecture, context candidates, preprocessing, seeds, early stopping, and refit rules;
- resource budgets;
- output, provenance, and metric contracts; and
- protection of F1/2025 and the downstream regime analysis.

No result may be used to add a feature, candidate, seed, asset, architecture, metric, or alternative preprocessing path to the core benchmark.

## 2. Prediction task and keys

Each model predicts the next observed trading session's asset-local log return:

```text
target_log_return = log(adjusted_close[target_date] / adjusted_close[origin_date])
```

The model output is `predicted_log_return`. No required model directly predicts a raw stock price. Its secondary directional prediction is derived from the same regression output:

```text
predicted_direction = 1 if predicted_log_return > 0 else 0
```

A predicted return equal to zero maps to direction 0. There is no separate direction classifier in the core benchmark.

Every sample is keyed by `(asset, origin_date, target_date)`, where `target_date` is the asset's next observed trading session after `origin_date`. Partition membership and fold membership are determined by `target_date`, never by row position or `origin_date` alone.

## 3. Assets and model-fitting policy

The target assets, in canonical aggregation order, are:

1. AAPL
2. MSFT
3. GOOGL
4. NVDA

SPY is a market-reference feature source and is not a prediction target.

Both families fit one separate regression model per target asset. Pooled multi-stock fitting is prohibited in the required benchmark. All four assets use identical feature definitions, candidate spaces, fold schedules, selection rules, and evaluation rules. Within a fold, the selected structural candidate is shared across all four assets, while model parameters, LightGBM best iterations, LSTM scalers, LSTM best epochs, and fitted model states remain asset-specific as defined below.

## 4. Frozen predictive feature contract

The predictive matrix contains exactly these columns in this order:

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

The following columns are identifiers, targets, audit fields, or filters and must not enter either model's predictive matrix:

- `asset`
- `origin_date`
- `target_date`
- `target_log_return`
- `target_direction`
- `spy_observed`
- `core_evaluation_eligible`

`core_evaluation_eligible` is only a row-eligibility flag. Both families operate only on eligible observations with finite target and feature values. No new predictive feature, missingness indicator, imputation path, feature selector, clipping rule, or transform may be introduced after learned-model results are seen. RSI, MACD, Bollinger Bands, fundamentals, news, sentiment, macroeconomic releases, categorical regime labels, and any additional technical indicator are outside the core benchmark.

All D1–D5 learned-model test predictions must cover exactly the canonical eligible Phase 2C baseline keys for the same `(fold, asset, origin_date, target_date)` combinations. Implementations must assert this key equality separately for LightGBM and for each LSTM seed before metrics are computed. Within one model/configuration/seed slice, a missing, extra, or duplicate test key is an error; the three intentional LSTM seed realizations remain distinct through the `seed` field. Coverage failures must not be hidden by taking an intersection or silently shrinking the evaluation set.

## 5. Frozen chronological folds

Calendar years below refer to `target_date`:

| Fold | Training targets | Validation targets | Test targets | Status during development |
|---|---|---|---|---|
| D1 | 2015–2018 | 2019 | 2020 | Permitted |
| D2 | 2015–2019 | 2020 | 2021 | Permitted |
| D3 | 2015–2020 | 2021 | 2022 | Permitted |
| D4 | 2015–2021 | 2022 | 2023 | Permitted |
| D5 | 2015–2022 | 2023 | 2024 | Permitted |
| F1 | 2015–2023 | 2024 | 2025 | Prohibited until the final-test gate |

Each fold is an independent causal selection exercise. Later-fold validation data, any test metric, D1–D5 aggregate test results, F1 data, and regime-specific results must not affect an earlier fold's candidate scores, winner, fitted state, or predictions. Model parameters remain fixed for the entire test year; newly observed feature inputs through an origin date may be used, but there is no parameter update, incremental fit, or test-year early stopping.

## 6. Fold-local candidate selection

Candidate spaces are global and fixed by this document. Candidate scores and winners are fold-local.

For each candidate and development fold:

1. Fit one candidate model per target asset on that fold's training partition only.
2. Use only that fold's validation partition for early stopping and candidate evaluation.
3. Compute validation MAE separately for AAPL, MSFT, GOOGL, and NVDA.
4. Give each asset equal weight and calculate `macro_validation_mae` as the arithmetic mean of the four asset MAEs.
5. Select one winning structural `config_id` for that family and fold.

The winning structural configuration is therefore common to the four asset models inside a fold. This does not pool their observations or fitted parameters.

For LSTM, the per-candidate asset MAE used in step 4 is first averaged equally over all three fixed seeds. The LSTM macro statistic is then averaged equally over the four asset-level seed means. This order prevents assets and seeds with more rows from receiving greater weight.

The sole primary selection statistic is the fold's equal-weight `macro_validation_mae`. Selection must not use test MAE, test RMSE, test directional accuracy, D1–D5 aggregate test performance, F1/2025, regime performance, runtime alone, or any future fold.

Candidate comparisons must be complete and like-for-like. Every candidate in a family must produce finite validation predictions and metrics for every required asset; every LSTM candidate must do so for every fixed seed as well. All candidates and seeds must cover the same eligible validation endpoint keys within an asset/fold. A missing, extra, duplicate, failed, or non-finite validation slice invalidates that candidate grid for the fold: no winner may be selected from the incomplete grid, and the benchmark must stop and report the failure. Intersections or selective omission are prohibited.

## 7. Frozen tie-breaking rule

A candidate is practically tied with the candidate having the lowest `macro_validation_mae` when:

```text
abs(candidate_macro_validation_mae - best_macro_validation_mae) <= 0.00001
```

The threshold is in log-return units and cannot change after results are observed. Resolve the complete practically tied set in this order:

1. choose the simpler/lower-compute structural candidate;
2. if structural simplicity is still tied, choose the lower macro validation RMSE;
3. if still tied, choose the lexically smallest `config_id` using ascending, case-sensitive string ordering.

Family-specific structural simplicity is fixed as follows:

- LightGBM: fewer `num_leaves` is simpler; if equal, larger `min_child_samples` is simpler because it permits fewer fine-grained leaves. Equivalently, sort by `(num_leaves ascending, min_child_samples descending)`.
- LSTM: the shorter `context_length` is simpler. `LSTM_21` therefore precedes `LSTM_63` on this criterion.

LightGBM macro validation RMSE is the equal-weight mean of the four asset validation RMSE values. LSTM macro validation RMSE first averages seed-specific validation RMSE equally within each asset and then averages the four asset means equally. Tie-break metrics are computed only from the same fold's validation predictions.

## 8. LightGBM preregistration

### 8.1 Framework and fixed settings

The classical learned benchmark uses LightGBM 4.x and `LGBMRegressor`, with one model per asset. Continuous features are not scaled or standardized.

| Setting | Frozen value |
|---|---:|
| objective | `regression_l1` |
| metric | `l1` |
| learning_rate | 0.03 |
| maximum boosting rounds / `n_estimators` | 1000 |
| early_stopping_rounds | 50 |
| feature_fraction | 1.0 |
| bagging_fraction | 1.0 |
| bagging_freq | 0 |
| lambda_l2 | 1.0 |
| verbosity | -1 |
| deterministic | `true` where supported |
| force_col_wise | `true` where supported |
| n_jobs | 1 for deterministic reference runs |
| random seed | 1729 |

The sklearn wrapper must receive `random_state = 1729`; no alternate seed value may be passed through a LightGBM seed alias or subordinate seed parameter. The implementation must record the complete resolved parameter set. Settings not enumerated in this preregistration use the defaults of the exact LightGBM 4.x patch version locked before the first fit. No test-period information may be used to change them.

### 8.2 Candidate grid

| `config_id` | `num_leaves` | `min_child_samples` |
|---|---:|---:|
| `LGBM_01` | 15 | 20 |
| `LGBM_02` | 31 | 20 |
| `LGBM_03` | 15 | 50 |
| `LGBM_04` | 31 | 50 |

These are the only LightGBM candidates. No candidate may be removed, added, or altered after training starts.

### 8.3 Candidate fit, early stopping, and refit

For each `(fold, config_id, asset)` candidate fit:

- fit model parameters on training observations only;
- provide that asset's validation observations as the sole `eval_set`;
- monitor validation L1/MAE;
- initialize the sklearn estimator with `n_estimators = 1000` and use `lightgbm.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False, min_delta=0.0)`, or an exactly equivalent LightGBM 4.x callback invocation;
- retain validation predictions from the selected early-stopping state; and
- record the asset-specific, positive one-based `best_iteration` in the inclusive range 1–1,000 and the validation metrics. If training reaches the 1,000-round cap without triggering the stopping callback, retain the iteration identified by LightGBM as the best validation-L1 iteration rather than assuming that the cap itself was best.

After the fold-level winning `config_id` is selected, each asset is refit independently on that fold's combined training-plus-validation observations. The refit uses the winning structural parameters and exactly that asset's previously observed `best_iteration` from the winning candidate's train-only/validation fit. It starts from a fresh model, performs no early stopping, never supplies the test year as an evaluation set, and uses no test metric or target. The refitted model is frozen for the entire test year.

The saved record for every refit must include the winning fold-level configuration, asset-specific best iteration, seed, train/validation/refit bounds, feature order, effective parameters, and fitted artifact identifier.

## 9. LSTM preregistration

### 9.1 Framework and architecture

The deep-learning benchmark uses PyTorch. TensorFlow is not permitted in the core implementation. A separate model is fitted for each asset and seed.

The input tensor shape is `(batch, context_length, 17)` (`batch_first = true`). The model contains:

```text
17-feature input
→ one unidirectional LSTM layer with hidden_size 32
→ final-timestep LSTM output
→ dropout with probability 0.20
→ linear layer with one scalar output
```

Because the LSTM has one recurrent layer, dropout is a separate post-LSTM module applied to the final-timestep representation; it is not PyTorch's inter-layer LSTM dropout. Dropout is enabled only during fitting and disabled for validation and prediction.

| Setting | Frozen value |
|---|---:|
| hidden_size | 32 |
| num_layers | 1 |
| post-LSTM dropout | 0.20 |
| output_size | 1 |
| optimizer | Adam |
| learning_rate | 0.001 |
| loss | MAE / L1 loss |
| batch_size | 64 |
| maximum epochs | 60 |
| early-stopping patience | 8 |
| early-stopping minimum delta | 0.00001 |
| gradient clip norm | 1.0 |
| shuffle training sequences | `false` |
| target scaling | none |
| bidirectional | `false` |
| attention | none |
| stacked/recurrent layers | none beyond the single layer |
| recurrent dropout | none |
| tensor/model floating-point dtype | `float32` |
| hidden/cell state policy | zero/reset for every independent sequence batch; no state carried across samples or batches |
| final training batch | retained (`drop_last = false`) |
| data-loader workers | 0 for the deterministic reference run |

No attention, bidirectionality, stacked LSTM, recurrent dropout, target scaling, or architecture expansion is permitted.

Adam uses `betas = (0.9, 0.999)`, `eps = 1e-8`, `weight_decay = 0`, and `amsgrad = false`. Model initialization otherwise uses the defaults of the exact PyTorch 2.x patch version locked before the first fit, after applying the required seed.

### 9.2 Context candidate grid

| `config_id` | `context_length` | Methodological question |
|---|---:|---|
| `LSTM_21` | 21 sessions | Approximately one trading month of context |
| `LSTM_63` | 63 sessions | Approximately one trading quarter of context |

These are the only core LSTM candidates.

## 10. LSTM sequence construction

For a sample with `origin_date = t`, the sequence contains the `context_length` consecutive asset-local observed-session feature rows ending at and including `t`. The sequence target is the explicitly keyed `target_log_return` for that asset's next observed trading session. Each timestep uses the same 17 features in the frozen order.

Sequence construction must obey all of the following:

- sort within one asset by timestamp before windowing;
- map the endpoint to its target by explicit asset/date keys, never by an unverified tail offset;
- include only feature values observable at or before each timestep;
- include no target value in the input tensor;
- never cross an asset boundary;
- never pad or synthesize missing history;
- never delete an intervening date merely to make a sequence contiguous; and
- require every feature value in the sequence to be finite.

If the available chronological history cannot supply the complete context, that training or validation sample is not constructible. This natural context warm-up does not authorize deletion of a canonical D1–D5 test key: because every development test period occurs after ample history, the implementation must fail before scoring if a full valid sequence cannot be produced for every canonical test key.

All eligible validation endpoint keys must likewise be constructible and identical across `LSTM_21`, `LSTM_63`, and all three seeds. Context-dependent loss of early training endpoints is permitted only where a complete history genuinely does not exist; training endpoint counts must be recorded for each context candidate and may not be equalized by padding, duplication, or deletion of otherwise valid later rows.

Context may cross a partition boundary backward in time. For example, early validation sequences may use preceding training-period feature rows, and early test sequences may use preceding training/validation feature rows. Later validation or test sequences may also use earlier feature rows from their own period because those inputs were observable by the current origin date. Targets in validation or test never enter model fitting or preprocessing merely because their associated feature rows provide historical context.

Calendar-year boundaries do not reset hidden context. Future context, target leakage, position-based cross-ticker windows, and sequences created by first filtering out intervening regime dates are prohibited.

## 11. LSTM preprocessing

Only the 17 continuous input features are standardized; targets are never standardized.

During candidate selection, each `(fold, asset)` scaler is fitted once on that asset's finite, eligible training feature rows only and is shared by all LSTM candidates and seeds for that fold and asset. Each canonical feature row contributes to the fitted moments once; overlapping occurrences of the same row across multiple sequences are not flattened and counted repeatedly. Validation values and any historical context used by validation sequences are transformed with this training-only scaler; no validation or test statistic enters it.

After the winning context configuration is selected, a new scaler is fitted for each asset using only that fold's combined finite, eligible training-plus-validation feature rows, counting each canonical row once rather than once per overlapping sequence occurrence. This refitted scaler transforms refit sequences and all test inputs, including newly observed test-year feature rows. It is frozen for the test year.

The scaler contract is feature-wise centering and scaling using population moments (`ddof = 0`), equivalent to scikit-learn `StandardScaler`. For each feature (j):

```text
z_j = (x_j - training_mean_j) / training_scale_j
```

If the fitted variance is exactly zero, set `training_scale_j = 1.0` while retaining and centering the feature. Features are never silently dropped or reordered. Non-finite input on a required evaluation key is an error rather than an invitation to fit an unregistered imputer.

Every saved scaler must record the ordered feature names, fitting partition and date bounds, row count, means, variances, effective scales, data/config identifiers, and an artifact hash or identifier.

## 12. LSTM seeds and candidate aggregation

Use exactly these seeds, in canonical order:

1. 1729
2. 2718
3. 31415

For every `(candidate, fold, asset)`, run all three seeds. No seed may be removed, replaced, or selectively suppressed because it performs poorly.

For LSTM candidate selection:

1. compute validation MAE separately for each `(asset, seed)`;
2. average the three seed MAEs equally within each asset;
3. average the four resulting asset means equally; and
4. use this macro value in the selection and tie-breaking procedure in Sections 6–7.

The selected context is shared by all assets in that fold. Results and fitted states for all three seeds of the winning context are retained. The core benchmark must not select the best seed or ensemble/average the three seeds' row-level predictions.

## 13. LSTM early stopping and refit

During candidate fitting, early stopping is performed independently for every `(fold, asset, config_id, seed)`:

- fit weights using training sequences only;
- compute epoch-level MAE/L1 on the corresponding validation sequences with the model in evaluation mode;
- treat an epoch as an improvement only when its validation MAE is more than `0.00001` below the tracked best value;
- stop after eight consecutive completed epochs without such an improvement, or after epoch 60;
- retain/restore the state at the tracked best validation epoch;
- use one-based epoch numbering; and
- record `best_epoch`, the best validation MAE, stopping reason, and epochs run.

Training batches retain chronological sample order because `shuffle_training_sequences = false`. Gradients are clipped to a maximum norm of 1.0 after backpropagation and before each optimizer step. Candidate validation predictions are generated from the restored best-epoch state.

The final partial training batch is retained (`drop_last = false`), the deterministic reference loader uses `num_workers = 0`, and hidden/cell state is reset rather than carried between independent samples or batches. Epoch validation MAE is the arithmetic mean of absolute errors over individual validation rows, not an unweighted mean of batch means. A non-finite loss, prediction, or selection metric is a failed run.

After selecting the fold-level winning context, refit each asset and each fixed seed as follows:

1. fit a fresh scaler on combined training-plus-validation rows only;
2. initialize a fresh model with that same seed;
3. train on combined training-plus-validation sequences for exactly the `best_epoch` previously recorded for that `(fold, asset, winning config_id, seed)`;
4. do not early-stop or inspect the test year;
5. freeze model and scaler parameters for the entire test year; and
6. generate row-level predictions for all canonical test keys.

The refit does not continue from candidate weights. The chosen epoch remains asset- and seed-specific, while the chosen context length remains fold-level and common across assets. Every final fitted model record must include its chosen epoch.

## 14. Neural determinism and failure policy

The implementation must attempt deterministic behavior wherever supported and must:

- seed Python's random generator;
- seed NumPy;
- seed PyTorch CPU;
- seed all CUDA devices if CUDA is used;
- request deterministic PyTorch algorithms where practical;
- disable nondeterministic cuDNN benchmarking and request deterministic cuDNN behavior when applicable;
- initialize a fresh model after setting the seed for each fit/refit;
- record device type and hardware-relevant details;
- record Python and package versions; and
- record whether strict determinism was achieved, including any known exception.

CPU execution must remain supported. GPU use is permitted but not required, and cross-device bit identity is not assumed. A deterministic reference run must never replace an inconvenient seed with a different one.

Before a family is fitted, one exact dependency lock and one execution-device policy must be recorded and applied consistently to every candidate and refit in that family. CPU and GPU runs must not be mixed selectively in response to validation or test performance.

If a fixed seed fails because of a reproducible technical error, retry that exact seed once from a clean initialization. Do not substitute another seed. If the retry fails, record both attempts and the exception transparently, mark the affected run incomplete, and stop the benchmark rather than silently computing headline results from fewer seeds.

## 15. Frozen resource budget

| Family | Candidate calculation | Maximum candidate fits | Required winning refits |
|---|---|---:|---:|
| LightGBM | 4 candidates × 5 development folds × 4 assets | 80 | 5 folds × 4 assets = 20 |
| LSTM | 2 candidates × 3 seeds × 5 development folds × 4 assets | 120 | 5 folds × 4 assets × 3 seeds = 60 |

The LSTM maximum is 60 epochs per candidate fit, with early stopping expected to reduce actual work. These counts cover D1–D5; F1 is not part of development execution. A retry of an identical failed seed is a logged technical retry, not a new search candidate.

Search budgets must not expand after results are inspected. If local compute is impractical, stop and document the constraint. Do not silently remove weak seeds, reduce only poorly performing folds, change candidates, shorten selected runs, or skip assets.

## 16. Learned-model artifact and provenance contract

Future learned-model prediction artifacts must retain these canonical columns in this exact order:

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

LightGBM rows use `model = lightgbm` and seed 1729. LSTM rows use `model = lstm` and retain a separate row for every fixed seed and observation; no core prediction ensemble is saved or scored as a model. Test predictions use `partition = test`. Direction is calculated from the saved continuous prediction, not from an independent classifier.

Until the dedicated regime-analysis phase, every regime field must contain exactly:

```text
not_labeled_pre_regime_analysis
```

Regime placeholders are not predictive inputs and are not available for model selection.

This learned-model placeholder intentionally differs from the existing Phase 2C baseline placeholder `not_labeled_phase2c`. Both values mean “regime not yet labeled,” are never treated as regime categories, and remain in their original immutable prediction artifacts. The later regime-analysis phase must create traceable, versioned joined artifacts rather than overwrite either set of frozen predictions.

Each candidate/refit run must preserve enough immutable or content-addressed provenance to reconstruct its outcome, including:

- source revision and run identifier;
- experiment specification version 1.1;
- raw and processed dataset hashes shown at the top of this document;
- environment/package versions and environment-lock hash when available;
- fold definition and train/validation/test date bounds;
- exact ordered feature list;
- effective model configuration, model family, asset, and seed;
- fitted preprocessing artifact identifier, or an explicit LightGBM no-scaling marker;
- best iteration or best epoch as applicable;
- model artifact identifier;
- prediction artifact identifier/hash;
- runtime/device and determinism status; and
- completion status and any warning, retry, or exception.

Successful artifacts must not be overwritten in place. Candidate validation summaries, fold-level selection tables, fitted scalers, fitted model states, canonical predictions, metric tables, and run manifests must be traceable to the same run and data/configuration identities.

## 17. Metric and reporting contract

Required D1–D5 development-test metrics are:

- MAE in log-return units;
- RMSE in log-return units; and
- directional accuracy.

Directional accuracy must be displayed with sample count, actual class balance (the fraction of rows with `actual_direction = 1`), and the existing direction-persistence baseline. The canonical matched baseline artifact is:

- path: `results/baselines/predictions/baseline_predictions_e09e4909e9af7b7836c5e0b2c8d7b320d2848d5da9b33b8c9492421e0a108bb4.csv`;
- SHA-256: `e09e4909e9af7b7836c5e0b2c8d7b320d2848d5da9b33b8c9492421e0a108bb4`;
- `run_id`: `phase2c_dev_9352da6b4340`;
- `model`: `zero_return__direction_persistence`; and
- `model_config_id`: `zero_return__direction_persistence_v1`.

Its zero-return component supplies the MAE denominator, and its direction-persistence component supplies the directional reference. Learned-model MAE is compared with zero-return persistence on exactly matched observation keys using:

```text
MAE Skill = 1 - (MAE_model / MAE_baseline)
```

The baseline metric must use the same fold, asset, and `(origin_date, target_date)` rows. Matching must be one-to-one, with no duplicate, missing, or extra keys and identical `actual_log_return` and `actual_direction`; an intersection may not be used to hide a mismatch. If matched baseline MAE is zero, MAE skill is undefined and must be reported as such rather than coerced to a number.

Report learned-model metrics per asset and fold and as equal-weight macro summaries across AAPL, MSFT, GOOGL, and NVDA. Macro metrics, including macro MAE skill, are arithmetic means of the four corresponding asset-level metrics; macro MAE skill is not recalculated as a ratio of macro MAEs. Directional comparison additionally reports `DA difference = DA_model - DA_direction_persistence` on the same rows. For development-fold summaries, report each D1–D5 value and use the median and interquartile range as the default descriptive fold-to-fold summary; any mean/standard deviation is supplemental descriptive variability, not inferential uncertainty.

For LSTM, compute every metric separately for each seed before any summary. Later headline reporting must retain per-seed results, the equal-weight arithmetic mean across the three seeds, population standard deviation (`ddof = 0`), minimum, and maximum for each reporting cell. These are descriptive seed summaries, not inferential uncertainty. Row-level predictions must not be averaged across seeds in the core benchmark.

MAE is the candidate-selection metric. RMSE is only the second tie-break after structural simplicity, and directional accuracy is not a selection metric. Do not introduce MASE unless a separate preregistered standard naïve-error denominator based solely on permitted training history is approved through change control. The zero-return baseline ratio must be called baseline-relative MAE or MAE skill, never MASE.

Baseline-relative skill against itself is not a meaningful learned-model statistic. No regime degradation statistic, confidence interval, trading return, or profitability claim belongs in the learned-model development phase.

## 18. Final-test protection

F1/2025 remains untouched throughout model development. D1–D5 test results may be summarized retrospectively but cannot retroactively alter their predictions and cannot select the F1 winner.

The eventual F1 procedure, when separately authorized after all required structures and outputs are frozen, is:

1. use the unchanged family candidate spaces and selection rule;
2. fit candidates on 2015–2023 training targets only;
3. select the fold-level candidate using 2024 validation targets only;
4. freeze the selected configuration and asset-specific LightGBM iterations or asset-and-seed-specific LSTM epochs;
5. refit on combined 2015–2024 history without accessing 2025 targets;
6. evaluate 2025 once; and
7. preserve the immutable predictions and evaluation manifest.

This document does not authorize that procedure now. Before the final-test gate, no implementation may inspect F1 predictions, F1 metrics, 2025 target-conditioned diagnostics, or 2025 regime performance. Mechanical presence of frozen 2025 rows in the raw or processed dataset does not permit performance access.

No F1 candidate fitting or selection—including use of 2024 as F1 validation—may run during D1–D5 development. The future F1 winner must be selected afresh under the frozen candidate grid and rule; it cannot be copied from D5 or chosen from D1–D5 aggregate test results.

## 19. Regime-analysis separation

Regime is a later evaluation dimension, not an input, training condition, routing rule, candidate-selection dimension, or tuning signal. During LightGBM and LSTM development:

- do not condition fitting on regime;
- do not select configurations or epochs by regime;
- do not inspect candidate regime performance;
- do not change features after regime inspection; and
- leave all three regime columns at the required placeholder value.

Regime labels may be causally joined only in the dedicated regime-analysis phase after row-level predictions have been frozen. That later analysis must not modify the saved forecasts.

For the pre-regime learned-model gate, the automated obligation is to verify that regime columns remain the fixed sentinel and are unused by preprocessing, training, selection, and metrics. The causal regime-threshold leakage tests required by `EXPERIMENT_SPEC.md` become a hard gate before any real regime label is generated, joined, or analyzed. No claim is made in Phase 2D that those later thresholds have already been implemented or tested.

## 20. PatchTST gate

**PatchTST status: CLOSED.**

Phase 2D does not preregister or implement PatchTST. Its optional gate may be considered only after LightGBM and LSTM are complete, both pipelines are reproducible, D1–D5 development results exist, and compute feasibility can be assessed. If the gate is opened later, its candidate space and compute budget must be frozen before its first run. No PatchTST dependency, pilot, result, or hidden preprocessing path is authorized by this document.

## 21. Dependency plan

No dependency is installed or changed by this preregistration. Planned additions for their later implementation phases are limited to:

- LightGBM: `lightgbm>=4,<5`
- LSTM: `torch>=2,<3`
- LSTM preprocessing: `scikit-learn>=1.4,<2`

TensorFlow, transformers, and PatchTST dependencies must not be added for the core LightGBM/LSTM benchmark.

The permitted version ranges are planning constraints, not an execution lock. Before the first fit in each implementation phase, exact resolved package versions, Python version, platform, and device policy must be captured in an immutable environment lock/run manifest and then held constant for that family's candidate fits and refits.

## 22. Change control and implementation gate

Before the first learned-model development run, the machine-readable `configs/model_search.yaml` must agree exactly with this document. Implementations must validate or test the frozen feature order, asset order, candidate identifiers and values, seeds, folds, output schema, final-test prohibition, and placeholder policy before fitting.

Any material correction before final-test access requires a versioned amendment with its rationale and affected experiments. After learned-model results have been observed, performance-motivated changes are prohibited. A necessary correction must be disclosed as a new exploratory specification and must not silently replace the preregistered benchmark.

The next authorized implementation phase is Phase 2E: implement and execute the preregistered LightGBM benchmark on D1–D5 only, preserving the frozen dataset, feature set, configuration grid, fold-local selection rules, artifact schema, and F1 prohibition. Phase 2E does not begin automatically from this document.
