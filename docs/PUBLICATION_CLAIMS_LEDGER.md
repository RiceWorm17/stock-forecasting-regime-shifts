# Publication Claims Ledger

## Purpose and evidence hierarchy

This document defines the scientific claims that may and may not be made publicly for **When Stock Forecasting Models Fail: Stress-Testing Robustness Under Market Regime Shifts**. It is a claims review of existing artifacts only. No model was trained, no prediction was regenerated, no experiment was rerun, and no scientific artifact was changed to create this ledger.

**Public-candidate availability note.** Every aggregate D1–D5 result cited for the primary public findings is included under `results/`, but the underlying market-data rows, row-level predictions, checkpoints, and scalers are not redistributed. The verification labels record checks completed against the preserved private archive; they do not imply that a public-clone user can reconstruct row-level metrics or retrain the historical models. The F1/2025 artifact paths retained below identify the private evidence used in the review, but those artifacts are deliberately absent from this public candidate. F1 remains `INVALID_STOP_NO_RERUN`; see `docs/F1_INVALID_DISCLOSURE.md`.

The evidence hierarchy is:

1. **Primary evidence:** independently verified D1–D5 development results for target years 2020–2024.
2. **Secondary evidence:** independently verified, descriptive D1–D5 regime analysis, subject to the frozen `n >= 30` rule and incomplete fold coverage.
3. **Exploratory evidence only:** preserved 2025 artifacts from the original F1 execution, which remains **`INVALID_STOP_NO_RERUN`**.
4. **Excluded claims:** statements not supported by the completed design, sample, verification status, or authorized metrics.

This hierarchy must be preserved in future README, portfolio, CV, report, presentation, or application materials.

## Metric and status conventions

- `MAE Skill = 1 - MAE_model / MAE_zero_return_baseline`. Positive values favor the learned model; negative values favor zero-return persistence.
- `DA difference = DA_model - DA_direction_persistence`. Positive values favor the learned model on next-day sign classification.
- Regime `MAE-skill degradation = MAE_Skill_stressed - MAE_Skill_reference`. **Negative values mean worse baseline-relative skill under stress; positive values mean better skill under stress.** The sign must not be reversed in public descriptions.
- Development macro values are equal-weight means across AAPL, MSFT, GOOGL, and NVDA.
- LSTM headline values are arithmetic means of metrics calculated separately for seeds 1729, 2718, and 31415; predictions were not averaged and no best seed was substituted.
- Five-fold medians, quartiles, ranges, and seed dispersion are descriptive statistics, not confidence intervals.

Verification labels used below:

- **VERIFIED:** independently reconstructed or checked against saved row-level artifacts within the frozen tolerance.
- **VERIFIED — DESCRIPTIVE:** numerically verified, but not supported by formal time-series inference or a causal design.
- **POST-FAILURE EXPLORATORY:** reproducible from preserved F1 artifacts after the invalid stop, but not a successful or confirmatory final-test result.
- **EXCLUDE:** not supported, contradicted, or outside the completed study.

---

## A. Primary verified D1–D5 development findings

All claims in this section concern four large U.S. stocks—AAPL, MSFT, GOOGL, and NVDA—under five annual expanding-window development folds with target years 2020–2024. They do not establish universal market behavior and do not inherit validity from, or confer validity on, the original F1/2025 execution.

### D-01 — Exact cross-model evaluation population

1. **Exact scientific claim.** Baseline, LightGBM, and each of the three LSTM seeds were evaluated on exactly the same 5,032 keyed D1–D5 observations, with identical actual returns and directions and without intersection-based sample shrinkage.
2. **Supporting artifact paths.**
   - `results/combined/combined_verification_manifest_393be15393847bdcc48d9482a0d72f32437f1024e85523088dbc3194c4eabb02.json`
   - `docs/audit/PHASE_2G_REPORT.md`
3. **Actual numerical evidence.** Baseline: 5,032 rows; LightGBM: 5,032 rows; LSTM: 5,032 rows per seed and 15,096 rows total. There were 5,032 unique `(fold, asset, origin_date, target_date)` keys. Maximum actual-return disagreement was `0.0`; actual directions matched exactly. Fold observation counts were D1 `1,012`, D2 `1,008`, D3 `1,004`, D4 `1,000`, and D5 `1,008`.
4. **Sample sizes and limitations.** Five calendar folds, four assets, daily next-session log returns. This is an integrity and comparability result, not evidence that any model forecasts well. Git revision identity was unavailable, so provenance relies on content hashes and manifests.
5. **Verification status.** **VERIFIED — PASS**, including pre-write and post-write canonical-key audits.
6. **README and CV suitability.** README: **yes**, as a methods/integrity statement. CV: **yes**, as evidence of exact-key, leakage-aware evaluation engineering.
7. **Exclusion decision.** **Keep.** Do not present it as a performance result.

### D-02 — LightGBM gains were small and fold-dependent

1. **Exact scientific claim.** The selected LightGBM models achieved positive macro MAE Skill against matched zero-return persistence in three of five development folds, but the gains were small and inconsistent across years and assets.
2. **Supporting artifact paths.**
   - `results/combined/development_macro_fold_comparison_042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6.csv`
   - `results/combined/baseline_win_loss_counts_0e5210dfbf806fec4a065e138e115524c317242ed3a7c6e1b2e8068b3e858684.csv`
   - `results/combined/cross_fold_summary_5d6eeac0aad1b87b735714f0238f711b97fa6abce5b77608e42b0c905d3daad8.csv`
3. **Actual numerical evidence.** Macro MAE Skill was D1 `+0.005889608896`, D2 `-0.002023499156`, D3 `-0.017863111880`, D4 `+0.006915085367`, and D5 `+0.006464567841`. Skill was positive in `12/20` asset/fold cells and negative in `8/20`. The five-fold median was `+0.005889608896`; IQR was `0.008488066997`.
4. **Sample sizes and limitations.** Five annual folds, 20 asset/fold cells, and 5,032 keyed observations. Macro skill is the equal-weight mean of asset-level skills, not `1 - macro_MAE_model / macro_MAE_baseline`. No dependence-aware significance test or confidence interval was performed.
5. **Verification status.** **VERIFIED.** Independent row-level reconstruction had maximum LightGBM discrepancy `1.4710455076283324e-15`, below the `1e-12` tolerance.
6. **README and CV suitability.** README: **yes**. CV: **yes**, only as “small, fold-dependent development gains,” not general superiority.
7. **Exclusion decision.** **Keep with qualification.** Exclude “LightGBM consistently outperformed persistence.”

### D-03 — The preregistered LSTM did not improve macro MAE

1. **Exact scientific claim.** The three-seed LSTM metric mean failed to beat matched zero-return persistence on macro MAE in every D1–D5 development fold.
2. **Supporting artifact paths.**
   - `results/combined/development_macro_fold_comparison_042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6.csv`
   - `results/combined/baseline_win_loss_counts_0e5210dfbf806fec4a065e138e115524c317242ed3a7c6e1b2e8068b3e858684.csv`
   - `results/combined/cross_fold_summary_5d6eeac0aad1b87b735714f0238f711b97fa6abce5b77608e42b0c905d3daad8.csv`
3. **Actual numerical evidence.** Seed-mean macro MAE Skill was D1 `-0.067564707761`, D2 `-0.024037888343`, D3 `-0.057713425262`, D4 `-0.021335427633`, and D5 `-0.021774505507`. It was positive in only `4/20` asset/fold cells and negative in `16/20`. The five-fold median was `-0.024037888343`.
4. **Sample sizes and limitations.** Three fixed seeds, 5,032 observations per seed, and 15,096 LSTM rows. These results evaluate one frozen LSTM architecture/search/training protocol; they do not establish that all LSTMs or all deep models are inferior.
5. **Verification status.** **VERIFIED.** Maximum LSTM metric-reconstruction discrepancy was `7.216449660063518e-16`.
6. **README and CV suitability.** README: **yes**. CV: **yes**, as a negative scientific result with the frozen-protocol scope stated.
7. **Exclusion decision.** **Keep with scope.** Exclude “LSTMs are generally worse for stock forecasting.”

### D-04 — D3/2022 was a shared learned-model failure fold

1. **Exact scientific claim.** In D3, whose test targets are in 2022, both learned models trailed their matched baselines on macro MAE and directional accuracy.
2. **Supporting artifact paths.**
   - `results/combined/development_macro_fold_comparison_042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6.csv`
   - `results/combined/development_fold_ranking_e13f0bd43d0b2f3ae80e41f8d0278e808a1cf04509b1eda8c1986e13d6826de9.csv`
3. **Actual numerical evidence.** For `n=1,004`, zero-return persistence MAE was `0.021334487022`; direction-persistence accuracy was `0.521912350598`. LightGBM had MAE `0.021654939710`, MAE Skill `-0.017863111880`, DA `0.471115537849`, and DA difference `-0.050796812749`. LSTM seed mean had MAE `0.022467453746`, MAE Skill `-0.057713425262`, DA `0.487383798141`, and DA difference `-0.034528552457`.
4. **Sample sizes and limitations.** One annual fold across four assets. This identifies a shared failure period but does not identify its cause or prove that a market regime caused the result.
5. **Verification status.** **VERIFIED.** Saved fold metrics and rankings were independently reproduced.
6. **README and CV suitability.** README: **yes**. CV: **yes**, if explicitly described as a 2022 development-fold result.
7. **Exclusion decision.** **Keep as descriptive evidence.** Exclude causal wording.

### D-05 — Return-error and directional conclusions were not interchangeable

1. **Exact scientific claim.** Better next-day sign classification sometimes occurred without positive MAE Skill, so regression-error and directional findings cannot be collapsed into one “best model” conclusion.
2. **Supporting artifact paths.**
   - `results/combined/development_macro_fold_comparison_042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6.csv`
   - `results/combined/development_asset_fold_comparison_61ad31f5e745338ccf679be9614188bab869964b350ab4f32a26067f85a2f3a1.csv`
   - `results/combined/baseline_win_loss_counts_0e5210dfbf806fec4a065e138e115524c317242ed3a7c6e1b2e8068b3e858684.csv`
3. **Actual numerical evidence.** LightGBM DA difference was positive in `4/5` macro folds and `14/20` asset/fold cells, while MAE Skill was positive in `3/5` folds and `12/20` cells. LSTM seed-mean DA difference was positive in `3/5` folds and `11/20` cells, although macro MAE Skill was negative in all five folds and asset/fold skill was negative in `16/20` cells. The signs of MAE Skill and DA difference disagreed in `4/20` LightGBM cells and `7/20` LSTM cells.
4. **Sample sizes and limitations.** Five folds, 20 asset/fold cells, and 5,032 keyed observations. Directional accuracy ignores return magnitude, turnover, transaction costs, position sizing, and risk; it is not evidence of profitability.
5. **Verification status.** **VERIFIED.** Both metrics were independently reconstructed from the same exact-key populations.
6. **README and CV suitability.** README: **yes**. CV: **yes**, as a robustness/evaluation finding.
7. **Exclusion decision.** **Keep.** Exclude any inference from directional accuracy to economic value.

### D-06 — LSTM conclusions were sensitive to initialization

1. **Exact scientific claim.** LSTM MAE-Skill conclusions changed sign across the three preregistered seeds in 8 of 20 asset/fold cells, demonstrating fixed-seed initialization sensitivity within the tested protocol.
2. **Supporting artifact path.** `results/combined/lstm_seed_stability_f965911f3b488f43e12e34996cef785e40de4d2d54fa52485f3e2f4064197d95.csv`
3. **Actual numerical evidence.** Sign changes occurred in D1 GOOGL and NVDA; D2 AAPL; D4 GOOGL and MSFT; and D5 AAPL, GOOGL, and NVDA. The largest MAE dispersion was D1/MSFT, with population SD `0.002069483353` and range `0.004804365505`. The smallest was D5/GOOGL, with population SD `0.000049675123` and range `0.000121122840`.
4. **Sample sizes and limitations.** Three seeds and 20 asset/fold cells. Population SD and range are descriptive diagnostics, not uncertainty estimates for all possible initializations.
5. **Verification status.** **VERIFIED.** The stability table was independently reconstructed and content-addressed.
6. **README and CV suitability.** README: **yes**. CV: **optional**, secondary to the core benchmark findings.
7. **Exclusion decision.** **Keep with fixed-seed scope.** Exclude “LSTM results are unstable under every initialization.”

### D-07 — Added model complexity did not guarantee lower development error

1. **Exact scientific claim.** Macro-MAE ranking changed across folds, and the more computationally expensive LSTM never ranked first under the frozen development protocol.
2. **Supporting artifact paths.**
   - `results/combined/development_fold_ranking_e13f0bd43d0b2f3ae80e41f8d0278e808a1cf04509b1eda8c1986e13d6826de9.csv`
   - `results/combined/model_complexity_summary_e5d4ce6f88a686663da21b188f86bae92614075580ac7722486ab0844f8c05d6.csv`
3. **Actual numerical evidence.** LightGBM ranked first on macro MAE in D1, D4, and D5; zero-return persistence ranked first in D2 and D3; LSTM seed mean ranked third in all five folds. The recorded reference runs used 100 LightGBM fits in about `15.55 s` and 180 LSTM fits in about `1,064.876 s`; the baseline required no fit.
4. **Sample sizes and limitations.** Five folds. Runtimes are single local reference executions, not hardware-normalized benchmarks; a general speed ratio or systems-performance claim is not justified.
5. **Verification status.** **VERIFIED.** Rankings, fit counts, and recorded reference runtime fields match the saved artifacts.
6. **README and CV suitability.** README: **yes**. CV: **yes**, phrased as “complexity did not guarantee lower development error.”
7. **Exclusion decision.** **Keep with runtime caveat.** Exclude a universal efficiency claim.

---

## B. Verified D1–D5 regime findings

Regime labels were derived from SPY and joined to already-frozen predictions for evaluation. The three dimensions overlap. These results are descriptive associations, not causal effects. A negative degradation value means worse baseline-relative MAE Skill in the stressed category.

### R-01 — Stress responses were heterogeneous rather than uniformly harmful

1. **Exact scientific claim.** Across reportable development asset/fold contrasts, neither learned model showed a uniform loss of baseline-relative MAE Skill under negative-trend, high-volatility, or transition observations.
2. **Supporting artifact paths.**
   - `results/regime/development/asset_fold_regime_degradation_278dc64486ce255773a578bf6c285d7831e4f846c156ffab2a08ca99d45495eb.csv`
   - `results/regime/development/regime_failure_pattern_summary_68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f.csv`
   - `docs/audit/PHASE_2J_REPORT.md`
3. **Actual numerical evidence.** LightGBM was worse under stress in trend `2/12`, volatility `7/16`, and transition `4/16` reportable contrasts; corresponding medians were `+0.0197725869`, `+0.0047359234`, and `+0.0085800342`. LSTM seed mean was worse in trend `2/12`, volatility `9/16`, and transition `8/16`; corresponding medians were `+0.0336546836`, `-0.0076179482`, and `+0.0068000639`.
4. **Sample sizes and limitations.** `88/120` primary asset/fold contrasts were reportable. Regime dimensions overlap, observations are serially dependent, and cells are not independent replications. No formal inference was performed.
5. **Verification status.** **VERIFIED — DESCRIPTIVE.** Saved labels, exact-key joins, cell metrics, degradation signs, and summaries were independently reproduced.
6. **README and CV suitability.** README: **yes**. CV: **yes**, as “heterogeneous, model-dependent stress responses.”
7. **Exclusion decision.** **Keep.** Exclude “regime shifts generally caused model failure.”

### R-02 — The LSTM showed a qualified high-volatility vulnerability

1. **Exact scientific claim.** LSTM seed-mean baseline-relative MAE Skill was lower in high- than low-volatility observations in 9 of 16 reportable asset/fold contrasts and 3 of 4 reportable macro-fold contrasts.
2. **Supporting artifact paths.**
   - `results/regime/development/asset_fold_regime_degradation_278dc64486ce255773a578bf6c285d7831e4f846c156ffab2a08ca99d45495eb.csv`
   - `results/regime/development/macro_regime_degradation_55c49a10739c7fcdc430ad506edf8d2c172d9843da4de043e82080e97663c80a.csv`
   - `results/regime/development/cross_fold_regime_summary_046a0816d7940d5cb3b4b90817f9353024c3d3e0a60cce715efe98e4efded6ca.csv`
3. **Actual numerical evidence.** The asset/fold median high-minus-low MAE-Skill degradation was `-0.0076179482`. The four-reportable-fold macro median was `-0.0103471222`; fold values were D1 `-0.0422817259`, D2 `+0.0167092226`, D4 `-0.0130405002`, and D5 `-0.0076537441`.
4. **Sample sizes and limitations.** Per-asset high/low counts were D1 `220/33`, D2 `144/108`, D4 `135/115`, and D5 `93/159`; D3 had `251/0`, so no contrast was available. Only four macro folds were reportable, and seed signs disagreed in `9/16` reportable asset/fold contrasts. This is not a complete five-fold summary.
5. **Verification status.** **VERIFIED — DESCRIPTIVE AND QUALIFIED.** The cross-fold artifact marks the summary `descriptive_incomplete`.
6. **README and CV suitability.** README: **yes**, with counts and incompleteness. CV: **yes**, only with “descriptive” or “qualified” wording.
7. **Exclusion decision.** **Keep with strong qualification.** Exclude “high volatility causes LSTM failure” and “LSTM always degrades in high volatility.”

### R-03 — Transition observations did not consistently damage either model

1. **Exact scientific claim.** Transition-versus-stable MAE-Skill effects were mixed rather than consistently harmful: LightGBM was worse in 4 of 16 reportable asset/fold contrasts and LSTM seed mean was worse in 8 of 16.
2. **Supporting artifact paths.**
   - `results/regime/development/asset_fold_regime_degradation_278dc64486ce255773a578bf6c285d7831e4f846c156ffab2a08ca99d45495eb.csv`
   - `results/regime/development/macro_regime_degradation_55c49a10739c7fcdc430ad506edf8d2c172d9843da4de043e82080e97663c80a.csv`
   - `results/regime/development/regime_failure_pattern_summary_68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f.csv`
3. **Actual numerical evidence.** LightGBM had 4 negative and 12 positive reportable asset/fold degradations, median `+0.0085800342`; its macro results had `1/4` negative, median `+0.0062118233`. LSTM had 8 negative and 8 positive asset/fold degradations, median `+0.0068000639`; its macro results had `2/4` negative, median `+0.0022871673`.
4. **Sample sizes and limitations.** Reportable transition/stable counts were D1 `91/162`, D2 `52/200`, D3 `57/194`, and D5 `35/217`. D4 had only three transition observations and was suppressed. Only four macro folds were reportable.
5. **Verification status.** **VERIFIED — DESCRIPTIVE.** No causal mechanism was tested.
6. **README and CV suitability.** README: **yes**, as an important negative result. CV: **usually no**, because the qualification is difficult to preserve in a short line.
7. **Exclusion decision.** **Keep in detailed documentation.** Exclude “transitions consistently damage both models.”

### R-04 — Negative-trend observations did not consistently worsen performance

1. **Exact scientific claim.** For each learned model, only 2 of 12 reportable negative-versus-positive-trend asset/fold contrasts had worse baseline-relative MAE Skill in negative-trend observations.
2. **Supporting artifact paths.**
   - `results/regime/development/asset_fold_regime_degradation_278dc64486ce255773a578bf6c285d7831e4f846c156ffab2a08ca99d45495eb.csv`
   - `results/regime/development/cross_fold_regime_summary_046a0816d7940d5cb3b4b90817f9353024c3d3e0a60cce715efe98e4efded6ca.csv`
   - `results/regime/development/regime_failure_pattern_summary_68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f.csv`
3. **Actual numerical evidence.** LightGBM had 2 negative and 10 positive asset/fold degradations, median `+0.0197725869`; all three reportable macro folds were positive, with median `+0.0244493776`. LSTM had 2 negative and 10 positive asset/fold degradations, median `+0.0336546836`; all three reportable macro folds were positive, with median `+0.0421451609`.
4. **Sample sizes and limitations.** Reportable negative/positive counts were D1 `64/189`, D3 `196/55`, and D4 `53/197`. D2 had only two negative-trend observations and D5 had none, so only three macro folds were reportable. The positive medians do not prove beneficial causal effects.
5. **Verification status.** **VERIFIED — DESCRIPTIVE AND INCOMPLETE.** Cross-fold status is `descriptive_incomplete`.
6. **README and CV suitability.** README: **yes**, with the three-fold limitation. CV: **no** as a standalone finding.
7. **Exclusion decision.** **Keep in detailed documentation.** Exclude “negative markets consistently cause forecasting failure” and any claim that negative trends improve forecasts causally.

### R-05 — Regime conclusions depended on the evaluation objective

1. **Exact scientific claim.** MAE-Skill degradation and directional-accuracy degradation gave opposing stress conclusions in 15 of 88 reportable asset/fold regime contrasts.
2. **Supporting artifact path.** `results/regime/development/directional_error_disagreement_9330d2595da96b6f87f3c32585326873ace2f6aab738c58dbf9d3604fa6c35eb.csv`
3. **Actual numerical evidence.** LightGBM had `5/44` disagreements: four direction-worse-only and one MAE-skill-worse-only. LSTM seed mean had `10/44`: three direction-worse-only and seven MAE-skill-worse-only. Total: `15/88`.
4. **Sample sizes and limitations.** The 88 contrasts already exclude low-`n` cells. MAE Skill and directional accuracy measure different properties; neither supports trading or profitability conclusions.
5. **Verification status.** **VERIFIED.** Disagreement flags and their inputs were independently reconstructed.
6. **README and CV suitability.** README: **yes**. CV: **yes**, as a robustness-analysis result.
7. **Exclusion decision.** **Keep.** Exclude a single-score or unqualified “best model” claim.

### R-06 — LSTM regime conclusions were seed-sensitive

1. **Exact scientific claim.** Across the three preregistered seeds, the sign of LSTM MAE-Skill degradation disagreed in 8 of 12 reportable trend, 9 of 16 volatility, and 6 of 16 transition asset/fold contrasts.
2. **Supporting artifact paths.**
   - `results/regime/development/lstm_per_seed_regime_cell_metrics_9305b93729681f279c8b17032ee5ef5dbee7c50f1390a9ddcb2409928945d0ac.csv`
   - `results/regime/development/lstm_seed_summary_regime_metrics_b2f81fde8a0a7f5ebc01068efd376096609ca5f5a671e3ef5a6e1c278b4265b8.csv`
   - `results/regime/development/regime_failure_pattern_summary_68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f.csv`
3. **Actual numerical evidence.** Asset/fold seed-sign disagreement counts were trend `8/12`, volatility `9/16`, and transition `6/16`. At macro level, disagreement occurred in trend `0/3`, volatility `1/4`, and transition `2/4` reportable contrasts.
4. **Sample sizes and limitations.** Three fixed seeds only. These are robustness diagnostics, not an estimated probability distribution over all possible seeds.
5. **Verification status.** **VERIFIED.** Per-seed cells, seed summaries, and failure counts were independently reproduced.
6. **README and CV suitability.** README: **yes**. CV: **yes**, if phrased as fixed-seed sensitivity rather than universal instability.
7. **Exclusion decision.** **Keep with scope.** Exclude “the seed-mean regime effect is robust across seeds.”

### R-07 — Sparse regime cells constrained headline conclusions

1. **Exact scientific claim.** The frozen `n >= 30` rule suppressed 32 of 120 primary asset/fold contrasts and 8 of 30 macro contrasts; none of the six model-by-dimension cross-fold summaries had complete five-fold support.
2. **Supporting artifact paths.**
   - `results/regime/development/development_regime_label_counts_45881ac07a118218bc4128e5e9975cd5dcc0410967ec491447fd3f9a2570ee26.csv`
   - `results/regime/development/cross_fold_regime_summary_046a0816d7940d5cb3b4b90817f9353024c3d3e0a60cce715efe98e4efded6ca.csv`
   - `results/regime/development/regime_failure_pattern_summary_68ba67ebb6f5f5ec2e801dee99d0345e9756d5b02c15613800e9c85d1d50391f.csv`
3. **Actual numerical evidence.** `88/120` asset/fold and `22/30` macro contrasts were reportable. Each model had three reportable trend folds and four reportable volatility and transition folds. Key sparse cells were D2 negative trend `n=2`, D3 low volatility `n=0`, D4 transition `n=3`, and D5 negative trend `n=0`.
4. **Sample sizes and limitations.** Suppressed values may remain audit evidence but cannot support headline degradation claims. Available fold medians are descriptive and incomplete, not confidence intervals.
5. **Verification status.** **VERIFIED.** Suppression states and counts followed the frozen preregistration.
6. **README and CV suitability.** README: **yes and mandatory as a limitation**. CV: **no** as a standalone claim.
7. **Exclusion decision.** **Keep as a mandatory limitation.** Exclude complete-five-fold and inferential claims.

---

## C. Verified methodological and reproducibility claims

### M-01 — Development metrics are independently reproducible

1. **Exact scientific claim.** Saved D1–D5 baseline, LightGBM, and LSTM metrics were independently recomputed from canonical row-level predictions and matched the saved metric tables to floating-point precision.
2. **Supporting artifact path.** `results/combined/combined_verification_manifest_393be15393847bdcc48d9482a0d72f32437f1024e85523088dbc3194c4eabb02.json`
3. **Actual numerical evidence.** Maximum absolute reconstruction differences were at most `1.4710455076283324e-15`, versus a `1e-12` tolerance. Baseline, LightGBM, and LSTM result trees contained 6, 113, and 235 files and retained their recorded tree identities.
4. **Sample sizes and limitations.** All 5,032 development keys and 15,096 LSTM seed rows were covered. Git commit identity was unavailable; provenance depends on content-addressed files, source-tree hashes, and manifests.
5. **Verification status.** **VERIFIED — PASS.** Input integrity, canonical-key integrity, and saved-table reconstruction passed.
6. **README and CV suitability.** README: **yes**. CV: **yes**, as independently verified, content-addressed evaluation engineering.
7. **Exclusion decision.** **Keep.** Do not claim that content hashes replace all benefits of a recorded Git revision.

### M-02 — Development regime evaluation respected the frozen information boundary

1. **Exact scientific claim.** D1–D5 regime labels used one unique SPY observation per origin date, and fold thresholds used same-fold training plus validation history only; no same-fold test or post-2024 observation entered threshold fitting.
2. **Supporting artifact paths.**
   - `docs/REGIME_PREREGISTRATION.md`
   - `configs/regime_analysis.yaml`
   - `results/regime/development/regime_threshold_audit_6f8ccb0a90c8bb9419061aa60cc6bcce781a26bcae5577f09cfa3a35920aa333.csv`
   - `results/regime/development/phase2j_independent_verification_d19ad9a5f41e904e591803b5c6f73617f7eafe27dcd9cd798f599c8e1f5573e1.json`
3. **Actual numerical evidence.** There were 1,258 unique `(fold, origin_date)` label rows; 5,032 labeled baseline rows; 5,032 labeled LightGBM rows; and 15,096 labeled LSTM rows. Independent verification recorded `same_fold_test_rows_included = 0` and `post_2024_rows_included = 0`. All 16 primary saved regime artifacts were reproduced with recorded maximum difference `0.0`.
4. **Sample sizes and limitations.** Five development folds only. Regime labels are evaluation dimensions, not model inputs. The labels support descriptive associations and do not create a causal experiment.
5. **Verification status.** **VERIFIED — PASS.** Threshold formulas, label joins, row counts, and metric layers were independently checked.
6. **README and CV suitability.** README: **yes**. CV: **yes**, as a leakage-aware evaluation-design contribution.
7. **Exclusion decision.** **Keep.** Exclude causal-regime and trading interpretations.

---

## D. 2025 preserved-artifact findings — original F1 remains INVALID

> **Mandatory disclosure before any 2025 number:** The original F1/2025 status is **`INVALID_STOP_NO_RERUN`**. The single authorized execution consumed its authorization and failed the frozen exact sealed-versus-canonical learned-prediction identity check after target access. A later read-only forensic audit identified a deterministic CSV parse-and-reserialize mechanism and reproduced the preserved tables, but did not convert the run to PASS. Every 2025 value below is a **post-failure exploratory finding**, not a successful, pristine, independently verified, or confirmatory final test. The verified D1–D5 development study remains the primary scientific evidence.

Historical integration tests had also accessed 2025 features and constructed/materialized 2025 targets. The forensic record found no evidence that those historical accesses produced earlier performance evaluation or methodology contamination, but 2025 must not be described as untouched or pristine.

### F1-01 — The original F1 execution is permanently invalid

1. **Exact scientific claim.** The original one-time F1 execution is permanently invalid, although its target-free prediction vectors passed the pre-access sealing checks.
2. **Supporting artifact paths.**
   - `results/final_test/state/FINAL_RUN_INVALID_3ccde4a2b7614a2e0aeb5dc4c12d4ecc871726bc9eb5d1354f796451e1a17a60.json`
   - `results/final_test/verification/sealed_prediction_vector_verification_60959c133a6f2d1c0b6f3b1d97acb2b3e808dbf23bb1e95ad910fffc09a22273.json`
   - `docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md`
3. **Actual numerical evidence.** Final state: `INVALID_STOP_NO_RERUN`; recorded failure: canonical learned predictions differed from sealed vectors; rerun prohibited. Pre-target verification covered 1,000 common asset/date keys, 1,000 baseline rows, 1,000 LightGBM rows, and 3,000 LSTM rows—1,000 for each seed—with finite predictions and no target columns.
4. **Sample sizes and limitations.** Four assets × 250 target dates produced 1,000 base keys. Pre-target sealing does not prove a fresh checkpoint-to-prediction replay. No normal final verification artifact or successful final run manifest exists.
5. **Verification status.** **VERIFIED PROCEDURAL FINDING.** The invalid status is immutable and was not changed by the forensic audit.
6. **README and CV suitability.** README: **mandatory whenever F1 is mentioned**. CV: **not a performance claim**; optional only as a reproducibility/governance lesson.
7. **Exclusion decision.** **Always retain the disclosure.** Exclude any statement that F1 passed, validated, replicated, or successfully completed the preregistered final test.

### F1-02 — A parse-and-reserialize cycle caused the exact-identity failure

1. **Exact scientific claim.** A deterministic CSV parse-and-reserialize cycle changed last-place learned-prediction representations and caused the frozen exact-equality verifier to reject the F1 run, without changing any forecast direction.
2. **Supporting artifact path.** `docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md`, Sections 8–10.
3. **Actual numerical evidence.** Baseline had 0 parsed mismatches. LightGBM had 293 parsed-value mismatches, 993 serialized-text mismatches, and maximum absolute difference `1.0001765041178778e-16`. LSTM had 1,124 parsed-value mismatches, 2,938 text mismatches, and maximum absolute difference `1.0061396160665481e-16`. Forecast-direction mismatches were 0.
4. **Sample sizes and limitations.** LightGBM had 1,000 rows; LSTM had 3,000 rows. Numerical immateriality does not override the frozen exact contract and does not restore validity. The audit did not rerun inference.
5. **Verification status.** **VERIFIED FORENSIC FINDING.** The mechanism reproduces the mismatch counts, but the run remains invalid.
6. **README and CV suitability.** README: **yes**, as a reproducibility-engineering finding. CV: **yes only as engineering work**, not as final-test success.
7. **Exclusion decision.** **Keep with the invalid-status disclosure.** Exclude “sealed and canonical predictions were exactly identical.”

### F1-03 — Exploratory overall 2025 performance

1. **Exact scientific claim.** In the preserved invalid-run 2025 artifacts, LightGBM had slightly lower equal-asset MAE than zero-return persistence, while the three-seed LSTM metric mean had higher MAE.
2. **Supporting artifact paths.**
   - `results/final_test/metrics/nonregime/f1_macro_direct_bdf87b913ff2904efa0a4d99e1d24cb201b3a7efc9bcdb4d814f6d671c1d58d3.csv`
   - `results/final_test/metrics/nonregime/f1_macro_lstm_seed_summary_dffa89cacadd5bec7ea306a4f00a65ff02c9bb0c86d1032c164fcdc7506b30e1.csv`
   - `docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md`
3. **Actual numerical evidence.** Persistence: MAE `0.014869915629502932`, RMSE `0.02176945271641622`, DA `0.478`. LightGBM: MAE `0.014821168884570853`, RMSE `0.021804041335958535`, DA `0.530`, MAE Skill `+0.002477400303526389`, DA difference `+0.05200000000000002`. LSTM seed mean: MAE `0.015237191286519658`, RMSE `0.022062339019920496`, DA `0.5133333333333333`, MAE Skill `-0.02841156731966385`, DA difference `+0.035333333333333335`.
4. **Sample sizes and limitations.** Four assets × 250 dates = 1,000 base observations; three LSTM seeds produced 3,000 LSTM rows. This is one calendar year, four stocks, and serially dependent data. There was no formal inference, fresh inference replay, or source-price reconstruction of the targets. The original run is invalid.
5. **Verification status.** **POST-FAILURE EXPLORATORY.** The forensic audit reproduced saved non-regime tables with maximum discrepancy `8.968520370800093e-16`; that does not equal an original verifier PASS.
6. **README and CV suitability.** README: **only in a separately titled exploratory appendix with the invalid disclosure repeated**. CV: **no**.
7. **Exclusion decision.** **Exclude from headlines, abstract-style summaries, and confirmatory claims.** It may appear only as explicitly post-failure exploratory evidence.

### F1-04 — Exploratory 2025 LightGBM performance varied by asset

1. **Exact scientific claim.** In the preserved 2025 artifacts, LightGBM MAE Skill was positive for AAPL, MSFT, and NVDA but negative for GOOGL.
2. **Supporting artifact path.** `results/final_test/metrics/nonregime/f1_asset_direct_8dea87c5314be5bb61bcb00765d56ff8cb55f8500f105fbb46e24d0bb649945a.csv`
3. **Actual numerical evidence.** AAPL `+0.0011019821`; MSFT `+0.0023050493`; NVDA `+0.0088393876`; GOOGL `-0.0023368177`.
4. **Sample sizes and limitations.** `n=250` per asset. There is no uncertainty interval, the sample covers one year, and all evidence comes from the invalid execution's preserved artifacts.
5. **Verification status.** **POST-FAILURE EXPLORATORY.** The values reproduce from saved rows but are not valid final-test findings.
6. **README and CV suitability.** README: **optional only in the exploratory appendix**. CV: **no**.
7. **Exclusion decision.** **Exclude from primary findings.** Do not generalize beyond these assets and dates.

### F1-05 — Exploratory 2025 regime associations

1. **Exact scientific claim.** Preserved 2025 artifacts show small negative LightGBM MAE-Skill degradation for transition and high-volatility contrasts, while LSTM seed-mean skill was lower under high volatility and negative trend; these are descriptive associations from an invalid run.
2. **Supporting artifact paths.**
   - `results/final_test/metrics/regime/f1_macro_regime_degradation_bc57c130d2a59de75c09732f7082eaa1238120024c934ab195366a1c91e0ba19.csv`
   - `results/final_test/regime/f1_unique_2025_regime_labels_6c07fee486542782690d36f48cabe9f8428f2a36eae0cb5f85fd28fb058215f0.csv`
   - `docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md`
3. **Actual numerical evidence.** LightGBM degradation was transition-minus-stable `-0.002970311470`, high-minus-low volatility `-0.002115952440`, and negative-minus-positive trend `+0.004538054854`. LSTM seed-mean degradation was transition `+0.000091334081`, volatility `-0.042396009238`, and trend `-0.028900402588`. The 250 unique origins comprised 67 transition/183 stable, 147 high/103 low volatility, and 64 negative/186 positive trend observations.
4. **Sample sizes and limitations.** 250 unique 2025 origins repeated across four assets; three LSTM seeds. Regimes overlap. The audit reproduced saved downstream tables to at most `1.8735013540549517e-15`, but it did not independently refit the 2015–2024 threshold distributions from source history. The execution remains invalid and no causal inference is permitted.
5. **Verification status.** **POST-FAILURE EXPLORATORY; downstream reproduction verified, end-to-end threshold reproduction partial.**
6. **README and CV suitability.** README: **only in the separately labeled exploratory appendix**. CV: **no**.
7. **Exclusion decision.** **Exclude from primary, confirmatory, and causal findings.** It cannot be used to say that F1 confirmed a development pattern.

---

## E. Unsupported or prohibited public claims

The following statements are not suitable for the README, CV, abstract, portfolio summary, presentation, or application materials.

| Proposed claim | Why unsupported or contradicted | Supporting exclusion evidence | README | CV | Decision |
|---|---|---|---|---|---|
| “LightGBM significantly outperformed persistence.” | Gains were small and mixed, and no dependence-aware statistical test or confidence interval was performed. | D-02; `results/combined/cross_fold_summary_5d6eeac0aad1b87b735714f0238f711b97fa6abce5b77608e42b0c905d3daad8.csv` | No | No | **EXCLUDE** |
| “LightGBM consistently outperformed persistence.” | Macro MAE Skill was negative in D2 and D3 and negative in 8/20 asset/fold cells. | D-02 | No | No | **EXCLUDE** |
| “LSTMs are universally inferior for stock forecasting.” | Only one frozen LSTM protocol, four assets, five development years, and three seeds were tested. | D-03, D-06 | No | No | **EXCLUDE** |
| “A regime shift caused model failure.” | Regime was an overlapping post-prediction evaluation dimension; the study was descriptive, not causal. | `docs/REGIME_PREREGISTRATION.md`; R-01 | No | No | **EXCLUDE** |
| “Transition periods consistently harmed both models.” | LightGBM was worse in 4/16 and LSTM in 8/16 reportable asset/fold contrasts; medians were positive. | R-03 | No | No | **EXCLUDE** |
| “Negative trends consistently harmed forecasting.” | Each model was worse in only 2/12 reportable asset/fold contrasts; only three macro folds were reportable. | R-04 | No | No | **EXCLUDE** |
| “High volatility always causes LSTM failure.” | LSTM was worse in 9/16 asset/fold and 3/4 macro contrasts, not all; seed signs disagreed in 9/16 cells. | R-02, R-06 | No | No | **EXCLUDE** |
| “Directional gains imply a profitable trading signal.” | Directional accuracy ignores magnitude, turnover, costs, execution, position sizing, and risk; no trading strategy was evaluated. | D-05; `docs/EXPERIMENT_SPEC.md` | No | No | **EXCLUDE** |
| “The models generated alpha” or were profitable/tradable. | PnL, Sharpe ratio, transaction costs, portfolio rules, and execution were explicitly outside scope. | `docs/EXPERIMENT_SPEC.md`; `docs/REGIME_PREREGISTRATION.md` | No | No | **EXCLUDE** |
| “The reported MAE Skill is MASE.” | The completed artifacts report matched-baseline MAE Skill, a different quantity. | Metric contract in `docs/REGIME_PREREGISTRATION.md` and combined tables | No | No | **EXCLUDE** |
| “The regime summaries cover all five folds.” | Every model/dimension summary is `descriptive_incomplete`; only three trend and four volatility/transition folds were reportable. | R-07 | No | No | **EXCLUDE** |
| “The five-fold quartiles or seed dispersion are inferential uncertainty.” | They are descriptive summaries, not confidence intervals or formal time-series inference. | `docs/REGIME_PREREGISTRATION.md`; D-06; R-07 | No | No | **EXCLUDE** |
| “PatchTST outperformed the completed benchmarks” or has D1–D5/F1 regime results. | No full PatchTST benchmark was admitted; pilot validation predictions were excluded from the regime study, and PatchTST is absent from F1. | `docs/audit/PHASE_2H_C_REPORT.md`; `docs/REGIME_PREREGISTRATION.md`; forensic audit | No | No | **EXCLUDE** |
| “F1/2025 passed, validated, replicated, or confirmed the development results.” | The original execution is `INVALID_STOP_NO_RERUN` and failed its frozen exact verification after target access. | F1-01; invalid-state artifact | No | No | **EXCLUDE** |
| “2025 was untouched or pristine.” | Historical integration tests accessed 2025 features and constructed/materialized targets; the original F1 then accessed targets. | `docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md` | No | No | **EXCLUDE** |
| “The saved F1 metrics passed the original independent verifier.” | The verifier stopped at exact sealed-versus-canonical learned-prediction inequality; no successful final verification artifact exists. | F1-01, F1-02 | No | No | **EXCLUDE** |
| “Fresh inference proved checkpoint-to-prediction identity for F1.” | The forensic audit did not rerun inference. | F1-01; forensic limitations | No | No | **EXCLUDE** |
| “The forensic audit independently reconstructed 2025 targets from prices or refit the F1 regime thresholds from source history.” | It verified internal alignment and preserved provenance but did neither source-level reconstruction. | F1-03, F1-05 | No | No | **EXCLUDE** |
| “No other F1 defect can exist.” | The audit found no additional conclusion-changing defect within bounded checks; that is not proof of absence. | `docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md` | No | No | **EXCLUDE** |
| “The local runtime gap establishes a general model-efficiency ratio.” | The recorded times come from one reference execution and are not hardware-normalized. | D-07 | No | No | **EXCLUDE** |

---

## F. Publication and portfolio recommendation

### Recommended primary public findings

The strongest defensible public narrative is:

1. Exact-key, leakage-aware expanding-window evaluation prevented sample mismatch and allowed independent row-level verification across 5,032 D1–D5 observations.
2. LightGBM produced only small, fold-dependent MAE gains over zero-return persistence: positive in three of five macro folds and 12 of 20 asset/fold cells.
3. The frozen three-seed LSTM protocol failed to beat persistence on macro MAE in every development fold and showed material seed sensitivity.
4. Return-error and directional conclusions frequently differed, so multiple objectives must be reported separately.
5. Regime responses were heterogeneous rather than uniformly harmful; the most defensible qualified stress result is the LSTM's descriptive high-volatility vulnerability.
6. Low regime counts and incomplete fold coverage materially limit headline regime conclusions.

### Recommended use of F1/2025

If 2025 evidence is retained publicly, place it in a distinct section titled **“Post-Failure Exploratory Analysis of Preserved 2025 Artifacts.”** Repeat the `INVALID_STOP_NO_RERUN` disclosure in that section. Do not place any F1 performance number in a headline, CV bullet, abstract-style summary, or statement of generalization.

### Outstanding scientific limitations

- Only four large U.S. equities and five development test years were studied.
- Daily observations and regime cells are serially dependent; no dependence-aware formal inference was preregistered or performed.
- Regime dimensions overlap and are descriptive evaluation slices, not causal treatments.
- The `n >= 30` rule removed 32/120 asset/fold and 8/30 macro regime contrasts from headline eligibility; no model/dimension had complete five-fold regime support.
- The LSTM evidence uses three fixed seeds and one frozen architecture/search protocol.
- Runtime is environment-specific and not a standardized systems benchmark.
- No trading, profitability, transaction-cost, risk, or portfolio analysis exists.
- PatchTST has no authorized full-benchmark finding and no F1 result.
- Git revision metadata was unavailable for the verified development benchmark; reproducibility relies on content hashes and manifests.
- The original F1 execution is permanently invalid; its preserved numbers are exploratory only, and source-level target reconstruction, threshold refitting, and fresh inference replay were not performed in the forensic audit.

## Final claims decision

Use the verified D1–D5 benchmark and its development regime analysis as the project's scientific core. Treat negative and mixed findings as results, not shortcomings to hide. Preserve the original F1 invalid status exactly, and use the 2025 artifacts only as a transparent reproducibility case study and, if desired, a separately labeled exploratory appendix.
