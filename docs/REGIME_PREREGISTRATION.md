# Regime Label Preregistration and Development Stress-Test Design

Status:  
**FROZEN BEFORE REGIME LABELING AND STRESS-TEST EXECUTION**

Scope:  
Defines market-regime labels, causal threshold rules, minimum-cell rules, model-comparison rules, and stress-test metrics before any regime-specific result is computed.

Authorized models for future regime analysis:

- persistence baseline;
- LightGBM; and
- LSTM seed-mean, with per-seed diagnostics retained.

PatchTST full benchmark:

- excluded from primary regime analysis because full-benchmark authorization was declined in Phase 2H-C; and
- PatchTST pilot validation predictions are not permitted in the regime study.

## 1. Authority and present-phase boundary

This Phase 2I preregistration operationalizes `EXPERIMENT_SPEC.md` version 1.1 without amending it. Its machine-readable counterpart is `configs/regime_analysis.yaml`. The two artifacts jointly freeze the development regime methodology before any regime label or regime-specific metric exists.

Phase 2I is design-only. It authorizes no regime-label generation, regime-specific calculation, model training or refitting, prediction mutation, historical-result mutation, PatchTST revival, F1/2025 access, trading analysis, or causal economic claim. The only Phase 2I outputs are this document, the YAML contract, and `PHASE_2I_REPORT.md`.

## 2. Trend tie-rule conflict and resolution

An initially proposed Phase 2I draft would have classified `Trend63_t >= 0` as `positive_trend` and `Trend63_t < 0` as `negative_trend`. That proposal contradicted the earlier frozen rule in `EXPERIMENT_SPEC.md` version 1.1, which classifies positive trend with `Trend63_t > 0` and negative trend with `Trend63_t <= 0`.

The conflict was resolved before any regime label, regime metric, or regime result was computed or observed. The decision is to preserve the earliest frozen rule:

- `positive_trend` if `Trend63_t > 0`;
- `negative_trend` if `Trend63_t <= 0`; and
- exact zero belongs to `negative_trend`.

No experiment-spec amendment or new experiment-spec version was created. Because the resolution preceded all labeling and analysis, it is not a data-dependent methodology change.

## 3. Scientific purpose

The project does not ask whether complex models always beat a simple baseline. Its central regime question is:

> Do learned forecasting models lose relative advantage against a simple zero-return persistence baseline under market-condition shifts?

The primary focus is baseline-relative MAE Skill, changes in that skill across regimes, disagreement between regression-error and directional-accuracy conclusions, and whether descriptive failures concentrate under particular market conditions. Regime is an evaluation dimension joined to already-generated predictions, not a model input, routing rule, training subset, or selection input.

Results may support descriptive association statements. They must not claim that a regime caused a model failure, establish a causal economic mechanism, or imply trading profitability.

## 4. Authoritative regime source and timestamp

SPY is the only regime source asset. It is the preregistered common market reference for AAPL, MSFT, GOOGL, and NVDA. No target-asset-specific or model-specific regime may be constructed.

Every label is evaluated at prediction `origin_date = t` using SPY observations available at or before that origin. A prediction's `target_date` assigns it to its chronological fold partition, but target-date SPY information must not enter its label. The same `(fold, origin_date)` label applies to every target asset, applicable model, and LSTM seed on that origin date.

Threshold distributions contain one finite SPY observation per unique origin date. The four repeated target-asset prediction rows must not give the same SPY date fourfold weight.

## 5. Primary regime dimensions

Exactly three overlapping primary regime dimensions are frozen:

1. trend;
2. volatility; and
3. transition.

They are reported separately rather than treated as mutually exclusive datasets or a primary joint-regime grid. No primary RSI, VIX, macro, news/sentiment, sector, or earnings regime may be added after results are seen.

## 6. Trend regime

At origin date `t`, define:

`Trend63_t = log(SPY_close_t / SPY_close_t_minus_63_observed_SPY_sessions)`

This is the already-created causal `spy_trend_63` feature. The frozen categories are:

- `positive_trend` when `Trend63_t > 0`;
- `negative_trend` when `Trend63_t <= 0`.

Exact zero is `negative_trend`. No threshold is fitted, no target-date return is used, and no future value is used. When 63 observed sessions are unavailable, the trend label is unavailable and the row is excluded from trend analysis only. It may remain eligible for volatility or transition analysis when those inputs exist. No trend label may be imputed or forward-filled.

## 7. Volatility regime

At origin date `t`, define annualized realized volatility:

`RV21_t = population_std(SPY_log_return_1d over the 21 observed SPY sessions ending at t, ddof=0, min_periods=21) * sqrt(252)`

The existing causal feature `spy_volatility_21` is the same population standard deviation before annualization. Multiplying both observations and their median threshold by `sqrt(252)` does not alter high/low classification.

For a D1–D5 development test partition, fit the fold threshold as the standard numeric median of finite `RV21` values from that same fold's combined training-plus-validation history. For an even number of values, the median is the arithmetic mean of the two middle order statistics. A same-fold test observation, a pooled D1–D5 test distribution, and 2025 are prohibited threshold inputs.

Classify:

- `high_volatility` when `RV21_t > fold_volatility_median_threshold`;
- `low_volatility` when `RV21_t <= fold_volatility_median_threshold`.

Equality belongs to `low_volatility`. If validation labels are later required, fit their threshold from training only. If an F1 test is eventually separately authorized, fit its threshold from F1 training plus validation only. Phase 2I performs neither operation.

An origin lacking the required 21-session input is unavailable for volatility analysis only. It is not imputed or globally removed.

## 8. Transition regime

At origin date `t`, define:

`RV21_jump_ratio_t = RV21_t / RV21_t_minus_21_observed_SPY_sessions`

and:

`abs_spy_return_t = abs(SPY_log_return_1d_t)`

The permitted-history shock threshold is the 95th percentile of finite `abs_spy_return` values. A row is labeled `transition` when either:

1. `RV21_jump_ratio_t >= 1.5`; or
2. `abs_spy_return_t > fold_abs_return_95th_percentile_threshold`.

Otherwise it is labeled `stable`. These canonical machine values correspond to the frozen experiment specification's semantic categories “Shifted” and “Stable/normal”; this naming alias changes no predicate.

A ratio exactly equal to 1.5 triggers `transition`. An absolute return exactly equal to the 95th-percentile threshold does not trigger the abnormal-return condition. If the ratio denominator is zero, unavailable, or non-finite—or the numerator is unavailable or non-finite—the ratio condition is false and no infinity is produced. A valid abnormal-return condition may still independently produce `transition`. If neither component has an available input, the transition label is unavailable rather than invented.

For D1–D5 test labels, estimate the return-shock threshold from finite same-fold training-plus-validation observations only. Validation would use training only; a future separately authorized F1 test would use its training-plus-validation history only.

## 9. Missing-history policy

Missing causal history affects only the regime dimension that requires it. It does not change the canonical prediction population or overall model-evaluation eligibility, and it does not delete a row from another dimension with valid inputs. Every unavailable count must be reported.

Regime values must never be invented, forward-filled, or imputed. Model sequences must remain intact; regime filtering happens only after labels are causally joined to unchanged prediction rows.

## 10. Permitted-history and fold policy

Fold partitions are assigned by the prediction row's `target_date`; threshold observations are SPY values at the corresponding `origin_date`. Only observed trading sessions are used.

| Fold | Train target dates | Validation target dates | Test target dates | Permitted history for test thresholds |
|---|---|---|---|---|
| D1 | 2015–2018 | 2019 | 2020 | D1 train + validation, 2015–2019 |
| D2 | 2015–2019 | 2020 | 2021 | D2 train + validation, 2015–2020 |
| D3 | 2015–2020 | 2021 | 2022 | D3 train + validation, 2015–2021 |
| D4 | 2015–2021 | 2022 | 2023 | D4 train + validation, 2015–2022 |
| D5 | 2015–2022 | 2023 | 2024 | D5 train + validation, 2015–2023 |

Thresholds are always same-fold. The same fold's test year, pooled D1–D5 test observations, later folds, and 2025 are prohibited. A validation analysis, if separately needed, uses the same fold's training partition only.

## 11. Quantile implementation

All threshold calculations first exclude and count non-finite source values. The volatility threshold uses the standard numeric median described in Section 7. The shock threshold is exactly equivalent to:

`pandas.Series.quantile(0.95, interpolation="linear")`

on finite permitted-history absolute SPY returns, with one observation per unique origin date. The interpolation method must not change after labels or results are seen. Cross-fold Q1 and Q3 summaries also use linear interpolation.

## 12. Label attachment and integrity

Labels join to prediction rows by `(fold, origin_date)`. The expected join is many prediction rows to one unique regime row. For LSTM, all three seed rows sharing `(fold, asset, origin_date, target_date)` receive the same regime label. The join must preserve prediction-row counts and keys; it may not use an intersection to hide missing, duplicate, or extra rows.

Regimes are not computed separately by target asset or model. Origin dates rather than target dates define information availability.

## 13. Sample-size and suppression rule

A regime cell is identified by model, fold, asset, seed status where applicable, regime dimension, and regime value. Its headline minimum is 30 prediction rows:

- `n >= 30`: status `reportable`;
- `n < 30`: status `descriptive_low_n`.

Every count is reported. A low-n cell may retain a numeric metric in audit or clearly labeled descriptive output, preserving the experiment specification's “descriptive only” rule, but it cannot support a headline degradation conclusion. The threshold cannot be lowered after results are observed.

LSTM seed diagnostics apply the minimum independently to each fold × asset × seed × regime cell. An LSTM seed-mean headline cell requires every contributing seed cell to be reportable. A headline macro contrast requires reportable cells for both compared regime values in all four assets.

## 14. Primary comparisons and fixed degradation signs

The three primary baseline-relative comparisons are:

- transition stress: `MAE_Skill_transition - MAE_Skill_stable`;
- volatility stress: `MAE_Skill_high_volatility - MAE_Skill_low_volatility`;
- trend stress: `MAE_Skill_negative_trend - MAE_Skill_positive_trend`.

The first-named category is the stressed category. A negative difference means the learned model has worse baseline-relative MAE Skill in that category. Signs must not be reversed after results are observed.

The frozen experiment specification also requires raw degradation alongside skill degradation. For each stressed/reference pair, report:

- `MAE_stressed - MAE_reference`;
- `RMSE_stressed - RMSE_reference`;
- `DA_stressed - DA_reference`; and
- directional stability loss, `DA_reference - DA_stressed`.

Positive MAE/RMSE degradation means higher error under stress; negative DA degradation and positive directional stability loss mean worse directional accuracy under stress. Raw degradation and MAE Skill degradation are complementary and neither substitutes for the other.

## 15. Model scope

Primary future reporting includes:

1. persistence baseline;
2. LightGBM; and
3. LSTM seed-mean.

LSTM per-seed values are diagnostic. PatchTST is excluded because Phase 2H-C declined its full benchmark and closed the optional challenger. PatchTST pilot validation predictions must not enter the development regime analysis.

The persistence artifact supplies two fixed references: zero-return persistence for regression and previous-direction persistence for directional accuracy. Regime outcomes cannot change any model, feature, target, prediction, or selection decision.

## 16. Metric contract and exact-key baselines

For each reportable learned-model regime cell, calculate:

- MAE;
- RMSE;
- directional accuracy;
- positive-direction class balance;
- zero-return baseline MAE on the exact same keys;
- MAE Skill;
- direction-persistence baseline directional accuracy on the exact same keys; and
- DA difference.

Definitions:

`MAE Skill = 1 - MAE_model / MAE_baseline`

`DA difference = DA_model - DA_direction_persistence`

References must match one-to-one on `(fold, asset, origin_date, target_date)` with identical actual return and direction. Intersection-only matching is prohibited. If matched baseline MAE is zero, MAE Skill is undefined and must not be coerced. Baseline-relative skill against the baseline itself is not a meaningful learned-model statistic; baseline cells report MAE, RMSE, directional accuracy, and positive class balance as references.

No PnL, trading-return, Sharpe-ratio, portfolio, transaction-cost, or return-simulation metric is authorized.

## 17. LSTM seed policy

LSTM row-level predictions are never averaged. Calculate each regime metric independently for seeds 1729, 2718, and 31415. Then, for each fold × asset × regime cell, summarize the three metric values using:

- arithmetic mean;
- population standard deviation with `ddof=0`;
- minimum; and
- maximum.

The arithmetic seed mean is the primary LSTM value; dispersion and per-seed values are diagnostic. No best-seed selection or prediction ensemble is allowed.

## 18. Macro aggregation

Macro values give equal weight to AAPL, MSFT, GOOGL, and NVDA. Assets are not weighted by row count.

Macro MAE Skill is the arithmetic mean of the four asset-level MAE Skill values. It must not be recomputed as `1 - macro_model_MAE / macro_baseline_MAE`. A headline macro regime contrast exists only when all four assets have reportable cells for both categories being compared.

For LSTM, calculate asset metrics per seed, then the equal-weight four-asset macro metric per seed, and only then summarize macro values across seeds.

## 19. Cross-fold descriptive summaries

Across D1–D5, use one reportable value per fold and report median, Q1, Q3, IQR, minimum, and maximum. Q1 and Q3 use linear interpolation. If fewer than all five fold values survive the frozen cell and macro rules, no complete headline cross-fold summary is reported; available values and their count may be shown descriptively.

These quantities are descriptive variability, not confidence intervals. Ordinary IID tests over daily rows are prohibited. Formal inference is outside scope unless separately preregistered with a method that respects time dependence.

## 20. Interpretation boundaries

Permitted language describes associations, for example that a model's MAE Skill is lower during transition observations than stable observations, or that its directional and regression conclusions disagree. It may also state that a model's skill is negative in a named reportable cell.

The analysis must not claim that a regime caused failure, that a strategy is profitable, that a model can trade through shifts, or that F1/2025 confirms a development pattern. Low-n output must remain explicitly descriptive.

## 21. Development and final-test boundary

Phase 2J is restricted to D1–D5 development test years 2020–2024. It must not access F1/2025. No 2025 row, label, metric, result, or placeholder is created in Phase 2I.

The accurate Phase 2I statement is:

> F1/2025 performance has not been evaluated, inspected, or used for model selection, methodology selection, or scientific conclusions. Phase 2I performs no new F1/2025 access.

This does not claim that historical row bytes were never materialized: Phase 2H-C already documented the closed PatchTST pilot loader's historical pre-filter materialization. That event produced no F1 performance evaluation or use and is not repeated here.

If F1 is eventually separately authorized, its regime thresholds use F1 training plus validation only, its model protocols and this methodology remain frozen, and development and final results remain clearly separated. F1 results cannot trigger a methodology change.

## 22. Artifact and execution policy

Phase 2I creates only:

- `docs/REGIME_PREREGISTRATION.md`;
- `configs/regime_analysis.yaml`; and
- `PHASE_2I_REPORT.md`.

It creates no label or metric artifact and changes no prediction or historical benchmark artifact. No baseline, LightGBM, LSTM, or PatchTST model is trained or refitted. Phase 2J must bind future outputs to this preregistration and preserve content-addressed provenance without overwriting the existing prediction artifacts.

## 23. Recommended next phase

After Phase 2I passes its consistency, test, and immutability gates, the recommended next phase is:

`Phase 2J / Development Regime Stress-Test Execution`

Phase 2J is not authorized or begun by this document; it requires a separate instruction and remains development-only.
