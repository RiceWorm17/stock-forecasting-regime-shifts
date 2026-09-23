# Phase 2J — Development Regime Stress-Test Execution

## 1. Phase status

**PHASE 2J DEVELOPMENT ANALYSIS COMPLETE.** All scientific artifacts were generated once from the frozen D1--D5 predictions, independently reloaded, and followed by a passing full test suite.

## 2. Frozen input integrity

PASS. The processed dataset, three canonical prediction files, frozen contracts, and six historical result trees matched their preregistered SHA-256 identities before and after artifact generation.

## 3. No-training confirmation

No model was fit, refit, optimized, or used to regenerate predictions. The Phase 2J source audit found zero training imports and zero training calls.

## 4. Development/F1 boundary

Only D1--D5 test targets from 2020--2024 were analyzed. The loader checked target_date before accessing retained SPY values, no post-2024 row entered an analysis population or artifact, and F1/2025 performance was not evaluated.

## 5. SPY source construction

The processed source supplied 10060 repeated target-asset rows and 2515 unique chronological SPY origin observations. Repeated SPY values and missingness agreed across all four target assets before deduplication; thresholds therefore gave each SPY date one vote.

## 6. Fold assignment

Partition membership used target_date, while every regime value used SPY information through origin_date only. An origin in the previous calendar year remained in the next target year's test fold when its target_date required it.

## 7. Threshold methodology

Each fold used its own training plus validation target-date population. Volatility used finite annualized population RV21 and its median; shock detection used the finite absolute SPY-return linear 95th percentile. Same-fold test, later-fold, pooled-test, and 2025 observations were excluded.

## 8. Fold threshold table

| fold | unique_permitted_origin_count | finite_rv21_count | volatility_median_threshold | finite_abs_return_count | abs_return_q95_threshold |
| --- | --- | --- | --- | --- | --- |
| D1 | 1257 | 1236 | 0.0990484 | 1256 | 0.0178354 |
| D2 | 1510 | 1489 | 0.106444 | 1509 | 0.0217704 |
| D3 | 1762 | 1741 | 0.107965 | 1761 | 0.0211022 |
| D4 | 2013 | 1992 | 0.120902 | 2012 | 0.0235634 |
| D5 | 2263 | 2242 | 0.122136 | 2262 | 0.0220071 |

## 9. Regime labeling implementation

Exactly 1258 unique fold/origin labels were created. Trend zero was negative, volatility equality was low, an RV jump ratio of 1.5 triggered transition, and return equality to q95 did not. Invalid jump denominators produced no infinity.

## 10. Trend counts

| fold | regime_value | count |
| --- | --- | --- |
| D1 | negative_trend | 64 |
| D1 | positive_trend | 189 |
| D1 | unavailable | 0 |
| D2 | negative_trend | 2 |
| D2 | positive_trend | 250 |
| D2 | unavailable | 0 |
| D3 | negative_trend | 196 |
| D3 | positive_trend | 55 |
| D3 | unavailable | 0 |
| D4 | negative_trend | 53 |
| D4 | positive_trend | 197 |
| D4 | unavailable | 0 |
| D5 | negative_trend | 0 |
| D5 | positive_trend | 252 |
| D5 | unavailable | 0 |

## 11. Volatility counts

| fold | regime_value | count |
| --- | --- | --- |
| D1 | high_volatility | 220 |
| D1 | low_volatility | 33 |
| D1 | unavailable | 0 |
| D2 | high_volatility | 144 |
| D2 | low_volatility | 108 |
| D2 | unavailable | 0 |
| D3 | high_volatility | 251 |
| D3 | low_volatility | 0 |
| D3 | unavailable | 0 |
| D4 | high_volatility | 135 |
| D4 | low_volatility | 115 |
| D4 | unavailable | 0 |
| D5 | high_volatility | 93 |
| D5 | low_volatility | 159 |
| D5 | unavailable | 0 |

## 12. Transition counts

| fold | regime_value | count |
| --- | --- | --- |
| D1 | transition | 91 |
| D1 | stable | 162 |
| D1 | unavailable | 0 |
| D2 | transition | 52 |
| D2 | stable | 200 |
| D2 | unavailable | 0 |
| D3 | transition | 57 |
| D3 | stable | 194 |
| D3 | unavailable | 0 |
| D4 | transition | 3 |
| D4 | stable | 247 |
| D4 | unavailable | 0 |
| D5 | transition | 35 |
| D5 | stable | 217 |
| D5 | unavailable | 0 |

## 13. Missing/unavailable counts

| fold | regime_dimension | regime_value | count |
| --- | --- | --- | --- |
| D1 | trend | unavailable | 0 |
| D1 | volatility | unavailable | 0 |
| D1 | transition | unavailable | 0 |
| D2 | trend | unavailable | 0 |
| D2 | volatility | unavailable | 0 |
| D2 | transition | unavailable | 0 |
| D3 | trend | unavailable | 0 |
| D3 | volatility | unavailable | 0 |
| D3 | transition | unavailable | 0 |
| D4 | trend | unavailable | 0 |
| D4 | volatility | unavailable | 0 |
| D4 | transition | unavailable | 0 |
| D5 | trend | unavailable | 0 |
| D5 | volatility | unavailable | 0 |
| D5 | transition | unavailable | 0 |

## 14. Sample-size audit

The fixed rule n >= 30 was applied. All 120 baseline, 120 LightGBM, and 120 primary LSTM cells were explicitly materialized, including n=0 cells. 32 of 120 asset/fold primary contrasts and 8 of 30 macro contrasts were not headline-reportable.

## 15. Baseline regime metrics

The baseline table contains 120/120 cells with MAE, RMSE, directional accuracy, positive-direction balance, n, and status. Baseline skill against itself was not calculated.

## 16. LightGBM regime metrics

The LightGBM table contains 120/120 cells. Every learned cell uses the exact saved Phase 2C baseline keys for MAE Skill and DA difference.

## 17. LSTM per-seed regime metrics

The LSTM per-seed table contains 360/360 cells for seeds 1729, 2718, and 31415. Predictions were never averaged.

## 18. LSTM seed-summary metrics

The LSTM summary contains 120/120 primary cells. Each metric was summarized after per-seed calculation using mean, population standard deviation, minimum, and maximum.

## 19. Transition degradation results

- lightgbm: 16 reportable asset/fold contrasts; 4 negative, 0 zero within tolerance, and 12 positive MAE-skill degradations; median 0.00858003.
- lstm_seed_mean: 16 reportable asset/fold contrasts; 8 negative, 0 zero within tolerance, and 8 positive MAE-skill degradations; median 0.00680006.

## 20. Volatility degradation results

- lightgbm: 16 reportable asset/fold contrasts; 7 negative, 0 zero within tolerance, and 9 positive MAE-skill degradations; median 0.00473592.
- lstm_seed_mean: 16 reportable asset/fold contrasts; 9 negative, 0 zero within tolerance, and 7 positive MAE-skill degradations; median -0.00761795.

## 21. Trend degradation results

- lightgbm: 12 reportable asset/fold contrasts; 2 negative, 0 zero within tolerance, and 10 positive MAE-skill degradations; median 0.0197726.
- lstm_seed_mean: 12 reportable asset/fold contrasts; 2 negative, 0 zero within tolerance, and 10 positive MAE-skill degradations; median 0.0336547.

## 22. Asset/fold headline contrasts

88 of 120 primary asset/fold contrasts met the two-sided n >= 30 rule. Non-reportable values remain descriptive and are not headline evidence.

## 23. Macro fold results

The macro table contains 180 rows, including baseline context, deterministic LightGBM, LSTM per-seed values, and LSTM seed summaries. 22 of 30 primary macro contrasts had complete four-asset support on both sides.

## 24. Cross-fold descriptive results

| model | regime_dimension | reportable_fold_count | median | q1 | q3 | complete_five_fold_summary | summary_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | trend | 3 | 0.0244494 | 0.0130814 | 0.0313854 | False | descriptive_incomplete |
| lightgbm | volatility | 4 | 0.00533624 | -0.00264493 | 0.0113722 | False | descriptive_incomplete |
| lightgbm | transition | 4 | 0.00621182 | 0.00353685 | 0.0126047 | False | descriptive_incomplete |
| lstm_seed_mean | trend | 3 | 0.0421452 | 0.0328491 | 0.0574581 | False | descriptive_incomplete |
| lstm_seed_mean | volatility | 4 | -0.0103471 | -0.0203508 | -0.001563 | False | descriptive_incomplete |
| lstm_seed_mean | transition | 4 | 0.00228717 | -0.0264661 | 0.0350512 | False | descriptive_incomplete |

## 25. Failure-pattern counts

| reporting_level | model | regime_dimension | reportable_contrasts | negative_degradation_count | zero_within_tolerance_count | positive_degradation_count | lstm_seed_sign_disagreement_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| asset_fold | lightgbm | trend | 12 | 2 | 0 | 10 | 0 |
| asset_fold | lightgbm | volatility | 16 | 7 | 0 | 9 | 0 |
| asset_fold | lightgbm | transition | 16 | 4 | 0 | 12 | 0 |
| asset_fold | lstm_seed_mean | trend | 12 | 2 | 0 | 10 | 8 |
| asset_fold | lstm_seed_mean | volatility | 16 | 9 | 0 | 7 | 9 |
| asset_fold | lstm_seed_mean | transition | 16 | 8 | 0 | 8 | 6 |
| macro_fold | lightgbm | trend | 3 | 0 | 0 | 3 | 0 |
| macro_fold | lightgbm | volatility | 4 | 1 | 0 | 3 | 0 |
| macro_fold | lightgbm | transition | 4 | 1 | 0 | 3 | 0 |
| macro_fold | lstm_seed_mean | trend | 3 | 0 | 0 | 3 | 0 |
| macro_fold | lstm_seed_mean | volatility | 4 | 3 | 0 | 1 | 1 |
| macro_fold | lstm_seed_mean | transition | 4 | 2 | 0 | 2 | 2 |

## 26. Directional/error disagreement

15 of 88 reportable asset/fold contrasts showed the preregistered XOR disagreement between worsening MAE Skill and worsening directional accuracy.

| model | fold | asset | regime_dimension | mae_skill_degradation | da_degradation | disagreement_type |
| --- | --- | --- | --- | --- | --- | --- |
| lightgbm | D1 | MSFT | trend | 0.00116837 | -0.0137235 | direction_worse_mae_skill_not_worse |
| lightgbm | D1 | MSFT | transition | 0.0164895 | -0.00474834 | direction_worse_mae_skill_not_worse |
| lightgbm | D2 | AAPL | transition | 0.0339385 | -0.0107692 | direction_worse_mae_skill_not_worse |
| lightgbm | D2 | NVDA | transition | 0.00598007 | -0.0192308 | direction_worse_mae_skill_not_worse |
| lightgbm | D3 | AAPL | transition | -0.0269596 | 0.0242358 | mae_skill_worse_direction_not_worse |
| lstm_seed_mean | D1 | AAPL | trend | -0.0268271 | 0.0424658 | mae_skill_worse_direction_not_worse |
| lstm_seed_mean | D1 | AAPL | volatility | -0.0241598 | 0.0232323 | mae_skill_worse_direction_not_worse |
| lstm_seed_mean | D1 | MSFT | trend | -0.00409424 | 0.0584215 | mae_skill_worse_direction_not_worse |
| lstm_seed_mean | D1 | MSFT | volatility | -0.0876882 | 0.00353535 | mae_skill_worse_direction_not_worse |
| lstm_seed_mean | D1 | MSFT | transition | 0.0232473 | -0.0398182 | direction_worse_mae_skill_not_worse |
| lstm_seed_mean | D3 | AAPL | transition | -0.0394586 | 0.00358715 | mae_skill_worse_direction_not_worse |
| lstm_seed_mean | D3 | GOOGL | transition | -0.0515125 | 0.00669199 | mae_skill_worse_direction_not_worse |
| lstm_seed_mean | D3 | NVDA | trend | 0.00910544 | -0.00717378 | direction_worse_mae_skill_not_worse |
| lstm_seed_mean | D4 | GOOGL | volatility | -0.0215791 | 0.00193237 | mae_skill_worse_direction_not_worse |
| lstm_seed_mean | D4 | NVDA | volatility | 0.0162867 | -0.0198604 | direction_worse_mae_skill_not_worse |

## 27. Low-n suppressed conclusions

32 asset/fold contrasts and 8 macro contrasts were suppressed from headline interpretation by the frozen sample-size rule. Their counts and descriptive numerical values remain in the audit artifacts.

## 28. Supported development conclusions

The following are descriptive associations, not causal claims:

- For lightgbm in the trend contrast, 2 of 12 reportable development asset/fold contrasts had worse stressed-regime baseline-relative MAE skill.
- For lightgbm in the volatility contrast, 7 of 16 reportable development asset/fold contrasts had worse stressed-regime baseline-relative MAE skill.
- For lightgbm in the transition contrast, 4 of 16 reportable development asset/fold contrasts had worse stressed-regime baseline-relative MAE skill.
- For lstm_seed_mean in the trend contrast, 2 of 12 reportable development asset/fold contrasts had worse stressed-regime baseline-relative MAE skill.
- For lstm_seed_mean in the volatility contrast, 9 of 16 reportable development asset/fold contrasts had worse stressed-regime baseline-relative MAE skill.
- For lstm_seed_mean in the transition contrast, 8 of 16 reportable development asset/fold contrasts had worse stressed-regime baseline-relative MAE skill.

## 29. Conclusions not supported

These results do not establish that a regime caused model failure, that any strategy is profitable, that the contrasts are statistically significant, that they generalize universally, or that F1/2025 confirms them.

## 30. Independent saved-file verification

PASS. All sixteen primary CSV artifacts were reloaded from disk after writing. Thresholds, labels, joins, cell metrics, exact baseline references, seed summaries, degradations, macro layers, cross-fold summaries, failure counts, and disagreement classifications reproduced within 1e-12.

## 31. Historical artifact immutability

PASS. Baseline, LightGBM, LSTM, Phase 2G combined, PatchTST pilot, and PatchTST authorization tree identities remained byte-identical. All seven frozen Markdown/YAML contracts retained their expected hashes.

## 32. Full test-suite result

PASS: **195/195 tests passed** in 21.24 seconds, with 0 failures, 0 skipped tests, and 0 warnings. This includes all original 159 tests plus 36 focused synthetic Phase 2J tests.

## 33. Artifact locations/hashes

- threshold_audit: results/regime/development/regime_threshold_audit_6f8ccb0a90c8bb9419061aa60cc6bcce781a26bcae5577f09cfa3a35920aa333.csv (SHA-256 6f8ccb0a90c8bb9419061aa60cc6bcce781a26bcae5577f09cfa3a35920aa333)
- unique_regime_labels: results/regime/development/development_unique_regime_labels_349f29371ff8d90bc0c57a0fce26292dbbdd5bd04ce6f224cb26b48d6d52a0cf.csv (SHA-256 349f29371ff8d90bc0c57a0fce26292dbbdd5bd04ce6f224cb26b48d6d52a0cf)
- regime_label_counts: results/regime/development/development_regime_label_counts_45881ac07a118218bc4128e5e9975cd5dcc0410967ec491447fd3f9a2570ee26.csv (SHA-256 45881ac07a118218bc4128e5e9975cd5dcc0410967ec491447fd3f9a2570ee26)
- labeled_baseline_predictions: results/regime/development/development_labeled_baseline_predictions_6dec7182317c643d6e0e41e9efd982eba9cdc257d0b8bb555b390128ee8dbb52.csv (SHA-256 6dec7182317c643d6e0e41e9efd982eba9cdc257d0b8bb555b390128ee8dbb52)
- labeled_lightgbm_predictions: results/regime/development/development_labeled_lightgbm_predictions_12ee398e53b239226387fe57994f3c07fef7044ce2fef0b99ff2f8de44cbb067.csv (SHA-256 12ee398e53b239226387fe57994f3c07fef7044ce2fef0b99ff2f8de44cbb067)
- labeled_lstm_predictions: results/regime/development/development_labeled_lstm_predictions_f9d354c87668903a11fb419321a671a811caf92e48dd74658fbf9a8d2d828a9d.csv (SHA-256 f9d354c87668903a11fb419321a671a811caf92e48dd74658fbf9a8d2d828a9d)
- baseline_regime_metrics: results/regime/development/baseline_regime_cell_metrics_e965db85359751403417205fa619584b6153d07f3854f342baa94e1bda12e0dc.csv (SHA-256 e965db85359751403417205fa619584b6153d07f3854f342baa94e1bda12e0dc)
- lightgbm_regime_metrics: results/regime/development/lightgbm_regime_cell_metrics_615e9464de3327d3d297fb718907bb71cdd5a6cf4de178d8afc3f0c1099b4c79.csv (SHA-256 615e9464de3327d3d297fb718907bb71cdd5a6cf4de178d8afc3f0c1099b4c79)
- lstm_per_seed_regime_metrics: results/regime/development/lstm_per_seed_regime_cell_metrics_9305b93729681f279c8b17032ee5ef5dbee7c50f1390a9ddcb2409928945d0ac.csv (SHA-256 9305b93729681f279c8b17032ee5ef5dbee7c50f1390a9ddcb2409928945d0ac)
- lstm_seed_summary_regime_metrics: results/regime/development/lstm_seed_summary_regime_metrics_b2f81fde8a0a7f5ebc01068efd376096609ca5f5a671e3ef5a6e1c278b4265b8.csv (SHA-256 b2f81fde8a0a7f5ebc01068efd376096609ca5f5a671e3ef5a6e1c278b4265b8)
- asset_fold_degradation: results/regime/development/asset_fold_regime_degradation_278dc64486ce255773a578bf6c285d7831e4f846c156ffab2a08ca99d45495eb.csv (SHA-256 278dc64486ce255773a578bf6c285d7831e4f846c156ffab2a08ca99d45495eb)
- macro_regime_metrics: results/regime/development/macro_regime_metrics_d892a985633c11f68bb500347ec6036166ce65e58df4544508357eeb9bcbe2fd.csv (SHA-256 d892a985633c11f68bb500347ec6036166ce65e58df4544508357eeb9bcbe2fd)
- macro_degradation: results/regime/development/macro_regime_degradation_55c49a10739c7fcdc430ad506edf8d2c172d9843da4de043e82080e97663c80a.csv (SHA-256 55c49a10739c7fcdc430ad506edf8d2c172d9843da4de043e82080e97663c80a)
- cross_fold_summary: results/regime/development/cross_fold_regime_summary_046a0816d7940d5cb3b4b90817f9353024c3d3e0a60cce715efe98e4efded6ca.csv (SHA-256 046a0816d7940d5cb3b4b90817f9353024c3d3e0a60cce715efe98e4efded6ca)
- failure_pattern_summary: results/regime/development/regime_failure_pattern_summary_68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f.csv (SHA-256 68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f)
- directional_error_disagreement: results/regime/development/directional_error_disagreement_9330d2595da96b6f87f3c32585326873ace2f6aab738c58dbf9d3604fa6c35eb.csv (SHA-256 9330d2595da96b6f87f3c32585326873ace2f6aab738c58dbf9d3604fa6c35eb)
- independent_verification: results/regime/development/phase2j_independent_verification_d19ad9a5f41e904e591803b5c6f73617f7eafe27dcd9cd798f599c8e1f5573e1.json (SHA-256 d19ad9a5f41e904e591803b5c6f73617f7eafe27dcd9cd798f599c8e1f5573e1)
- run_manifest: results/regime/development/phase2j_run_manifest_6e0aa516b2a5e21918cf5c6f97c641ac42ca54097865602867d27c6ff0647759.json (SHA-256 6e0aa516b2a5e21918cf5c6f97c641ac42ca54097865602867d27c6ff0647759)

## 34. Specification deviations

None. The write-free dry run caught one representation-only implementation bug: equal target timestamps loaded at different NumPy resolutions failed a strict Series dtype comparison. The check was corrected to compare normalized nanosecond timestamp values before execution. No scientific rule, threshold, label, metric, or result changed.

## 35. Remaining risks

Low-n transition cells can limit complete macro or five-fold conclusions. The three regime dimensions overlap and are descriptive views rather than independent causal mechanisms. Expanding fold histories make thresholds fold-specific by design, and LSTM seed dispersion can complicate a seed-mean result.

## 36. Recommended next phase

**Phase 2K / Development Regime Results Audit and Final-Test Authorization Review.** Phase 2K was not started.
