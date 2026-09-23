# When Stock Forecasting Models Fail

## A leakage-aware walk-forward study of robustness under market regime shifts

This project evaluates next-trading-day log-return forecasts for AAPL, MSFT, GOOGL, and NVDA under five expanding-window development folds covering 2020–2024, with SPY used only as a common market reference. Zero-return persistence, LightGBM, and a three-seed LSTM were compared on exactly matched prediction keys. Across 5,032 development observations, LightGBM achieved small, fold-dependent gains over persistence: macro MAE Skill was positive in three of five folds and 12 of 20 asset/fold cells. The frozen LSTM protocol had negative macro MAE Skill in every fold and in 16 of 20 asset/fold cells. Error-based and directional conclusions sometimes disagreed, while regime responses were heterogeneous and limited by sparse cells. These are descriptive forecasting results—not evidence of statistical significance, causality, trading profitability, or universal model superiority.

The scientific claims in this README are drawn from the independently checked [publication claims ledger](docs/PUBLICATION_CLAIMS_LEDGER.md). The verified D1–D5 development study is the scientific core. The separate F1/2025 execution remains **`INVALID_STOP_NO_RERUN`** and is not used to support the conclusions below.

This public release candidate is intentionally narrower than the preserved private research archive. It includes source code, frozen development specifications, aggregate D1–D5 evidence, and the four deterministic figures. It excludes provider-sourced market-data rows with unresolved redistribution rights, row-level predictions, trained checkpoints, scalers, final-test artifacts, course materials, and the original submission archive. See the [public file manifest](docs/PUBLIC_FILE_MANIFEST.md), [reproducibility boundary](docs/PUBLIC_REPRODUCIBILITY.md), and [license status](docs/LICENSE_STATUS.md).

**Development and rights disclosure.** The repository owner developed this project with AI assistance and is responsible for the final design, code, analysis, and documentation presented here. Dependencies and research-method inspirations are credited in [attribution and provenance](docs/ATTRIBUTION_AND_PROVENANCE.md). The repository intentionally has **no project license** and is not presented as open source; public visibility does not itself grant permission to reuse, modify, or redistribute the project files.

## 1. Research question

Do learned next-day return models retain baseline-relative predictive skill across future calendar years, assets, market conditions, random seeds, and evaluation objectives?

The study treats model failure as a measurable outcome. It asks where a learned model ceases to improve on a simple baseline, whether that conclusion changes between error and direction metrics, and whether stressed market categories reveal consistent degradation.

## 2. Main contributions

- Exact-key, chronological walk-forward evaluation with a separately frozen model-selection protocol.
- Fold-local configuration selection using validation MAE only; test outcomes, later folds, regime outcomes, and F1 were excluded from development selection.
- Matched zero-return and previous-direction persistence baselines.
- Seed-aware LSTM evaluation without best-seed substitution or forecast ensembling.
- Post-prediction SPY regime analysis with causal threshold histories and a frozen minimum-cell rule.
- Content-addressed saved artifacts plus independent row-level metric reconstruction.
- Explicit separation of verified development evidence from an invalid final execution.

The full frozen methods are in the [experiment specification](docs/EXPERIMENT_SPEC.md), [model preregistration](docs/MODEL_PREREGISTRATION.md), and [regime preregistration](docs/REGIME_PREREGISTRATION.md).

## 3. Dataset and prediction target

The target assets are **AAPL, MSFT, GOOGL, and NVDA**. **SPY is a market reference, not a fifth prediction target.** The source snapshot contains daily, auto-adjusted market data beginning in 2015. Each supervised row predicts the asset's adjusted-price log return for its next observed trading session:

$$
y_{t+1}=\log(P_{t+1}/P_t).
$$

The primary task is regression. Direction is derived from the same forecast (`predicted_log_return > 0`); no separate classifier is trained. Each learned model uses 17 preregistered asset and SPY features available no later than the forecast origin. The canonical key is `(asset, origin_date, target_date)`, and fold assignment is based on `target_date`. See the [data contract](docs/DATA_CONTRACT.md) and [data configuration](configs/data.yaml).

## 4. Leakage-aware methodology

For each development fold, the workflow:

1. fits candidate models on that fold's training partition;
2. evaluates candidates on the immediately following validation year;
3. chooses one structural configuration per model family using equal-weight macro validation MAE across the four assets;
4. refits a fresh per-asset model on combined training and validation rows; and
5. evaluates the frozen refit once on the next calendar-year test partition.

Feature scaling for the LSTM is fitted only on permitted feature rows. Test targets never participate in scaling, fitting, early stopping, configuration selection, or regime-threshold estimation. Models remain fixed throughout each test year. Learned predictions and baselines must match one-to-one on the complete canonical key set; intersection-only evaluation is prohibited.

## 5. Walk-forward fold design

The development study uses expanding training windows and one-year validation and test partitions:

| Fold | Training target dates | Validation | Development test | Test rows |
|---|---|---|---|---:|
| D1 | 2015–2018 | 2019 | 2020 | 1,012 |
| D2 | 2015–2019 | 2020 | 2021 | 1,008 |
| D3 | 2015–2020 | 2021 | 2022 | 1,004 |
| D4 | 2015–2021 | 2022 | 2023 | 1,000 |
| D5 | 2015–2022 | 2023 | 2024 | 1,008 |

The row counts pool four equal-length asset slices; reported macro metrics give each asset equal weight. Exact date boundaries are preserved in [configs/walk_forward.yaml](configs/walk_forward.yaml).

## 6. Baseline and learned models

- **Persistence baseline:** predicts a zero next-day log return for MAE/RMSE comparisons and carries the previous observed direction for directional comparison.
- **LightGBM:** separate deterministic `LGBMRegressor` models per asset, with four preregistered tree structures, validation-only early stopping, and one seed (`1729`).
- **LSTM:** separate one-layer PyTorch LSTMs per asset, using 17 features, hidden size 32, 21- or 63-session contexts, and three fixed seeds (`1729`, `2718`, `31415`). Metrics are computed per seed and then averaged; row-level predictions are not averaged and no best seed is selected.

A limited PatchTST pilot was used only for a preregistered feasibility gate. Authorization for a full PatchTST benchmark was declined, so there is no comparable D1–D5 PatchTST result and none is claimed here.

## 7. Model selection

Selection is local to each model family and fold. The statistic is equal-weight macro validation MAE across AAPL, MSFT, GOOGL, and NVDA.

- LightGBM compares four frozen structural candidates (`LGBM_01`–`LGBM_04`).
- LSTM compares 21- and 63-session contexts across all three seeds.
- Practical ties within `0.00001` MAE are resolved by a frozen simplicity rule, then RMSE, then configuration ID.
- The winning structure is shared across assets within the fold, while fitted asset models remain separate.
- Development test results, regime results, and 2025 outcomes cannot influence selection.

Selected configurations and refit audit trails are available in [results/lightgbm/selections](results/lightgbm/selections) and [results/lstm/selections](results/lstm/selections).

## 8. Market regime definitions

Regimes are overlapping, SPY-based evaluation dimensions attached **after** forecasts were frozen; they are not model inputs.

| Dimension | Stressed category | Reference category | Definition |
|---|---|---|---|
| Trend | Negative trend | Positive trend | Sign of the trailing 63-session SPY log trend |
| Volatility | High volatility | Low volatility | 21-session realized volatility relative to the same fold's train-plus-validation median |
| Transition | Transition | Stable | A 21-session volatility jump ratio of at least 1.5 or an origin-day absolute SPY return above the permitted-history 95th percentile |

For development tests, fitted thresholds use only the same fold's training and validation history. A contrast is headline-reportable only when both categories contain at least 30 observations. The primary quantity is

$$
\Delta\text{MAE Skill}=\text{Skill}_{stressed}-\text{Skill}_{reference},
$$

so **negative degradation means worse baseline-relative skill under stress**.

## 9. Verified development results

MAE Skill is defined as `1 − model MAE / matched zero-return-baseline MAE`; positive values favor the learned model. The saved macro results are:

| Fold (year) | n | LightGBM MAE Skill | LSTM seed-mean MAE Skill |
|---|---:|---:|---:|
| D1 (2020) | 1,012 | +0.005890 | −0.067565 |
| D2 (2021) | 1,008 | −0.002023 | −0.024038 |
| D3 (2022) | 1,004 | −0.017863 | −0.057713 |
| D4 (2023) | 1,000 | +0.006915 | −0.021335 |
| D5 (2024) | 1,008 | +0.006465 | −0.021775 |

LightGBM skill is positive in **3/5 macro folds** and **12/20 asset/fold cells**. The LSTM seed mean is negative in **all five macro folds** and **16/20 asset/fold cells**. D3/2022 is a shared failure fold: the zero-return baseline has MAE `0.021334`; LightGBM has MAE `0.021655`, MAE Skill `−0.017863`, and directional-accuracy difference `−0.050797`; the LSTM seed mean has MAE `0.022467`, MAE Skill `−0.057713`, and directional-accuracy difference `−0.034529` (`n = 1,004`).

Metric choice matters. MAE-Skill and directional-accuracy-difference signs disagree in **4/20 LightGBM** and **7/20 LSTM** asset/fold cells. LSTM MAE-Skill signs also change across seeds in **8/20** cells, so a single favorable seed would not represent the frozen protocol.

The macro values above come from the [development macro comparison CSV](results/combined/development_macro_fold_comparison_042b08dfe55c6449331f1ee5b9e8437a8540505c5061f61648d66c8d588203f6.csv); asset-level and seed diagnostics are in the [asset/fold comparison](results/combined/development_asset_fold_comparison_61ad31f5e745338ccf679be9614188bab869964b350ab4f32a26067f85a2f3a1.csv) and [LSTM seed-stability table](results/combined/lstm_seed_stability_f965911f3b488f43e12e34996cef785e40de4d2d54fa52485f3e2f4064197d95.csv).

## 10. Figures and interpretation

### Fold-level performance

![Grouped bars of LightGBM and LSTM macro MAE Skill for development folds D1 through D5](docs/figures/macro_mae_skill_by_fold.svg)

**Figure 1.** Equal-weight macro MAE Skill for the 2020–2024 development folds. Positive values favor the learned model. Values are descriptive summaries across four assets, not confidence intervals or significance tests.

### Asset/fold heterogeneity

![Two heat maps of asset-by-fold MAE Skill for LightGBM and LSTM](docs/figures/asset_fold_mae_skill.svg)

**Figure 2.** Asset-level MAE Skill across 20 cells per model. LightGBM is positive in 12/20 cells; the LSTM seed mean is positive in 4/20 and negative in 16/20. LSTM metrics were calculated separately by seed before averaging. The shared color scale and printed values preserve cross-model comparability.

### Regime degradation

![Reportable stressed-minus-reference MAE-Skill degradation points and medians by regime dimension](docs/figures/regime_mae_skill_degradation.svg)

**Figure 3.** Reportable asset/fold stressed-minus-reference changes in MAE Skill. LightGBM is worse under stress in `2/12` trend, `7/16` volatility, and `4/16` transition contrasts, with medians `+0.019773`, `+0.004736`, and `+0.008580`. The LSTM seed mean is worse in `2/12`, `9/16`, and `8/16`, with medians `+0.033655`, `−0.007618`, and `+0.006800`. Dimensions overlap and observations are serially dependent; these are descriptive associations, not causal effects.

### Incomplete regime coverage

![Stacked reportable and suppressed regime contrasts at asset-fold and macro-fold levels](docs/figures/regime_reportability.svg)

**Figure 4.** Under the frozen `n ≥ 30` rule, 88/120 model-level asset/fold contrast slots and 22/30 model-level macro/fold slots are reportable. Each model has only three reportable trend folds and four reportable volatility and transition folds. No model-by-dimension summary has complete five-fold macro support; suppressed cells are missing evidence, not zero effects. Because both models use the same regime populations, these slots should not be mistaken for unique independent samples.

Figure provenance, exact source paths, and the standard-library renderer are documented in [docs/figures](docs/figures/README.md).

## 11. What the results support

The evidence supports a narrow conclusion: under this frozen protocol, simple persistence is a strong benchmark, LightGBM's improvements are small and fold-dependent, and the evaluated LSTM configuration is usually worse on MAE. Stress responses are heterogeneous. The clearest qualified pattern is LSTM weakness in high-volatility contrasts—negative degradation in 9/16 reportable asset/fold cells and 3/4 reportable macro folds—but seed disagreement and incomplete coverage limit generalization. Across all regime dimensions, MAE and directional degradation disagree in 15/88 reportable contrasts.

The evidence does **not** establish statistical significance, causal regime effects, trading profitability, universal superiority of one model family, or a validated 2025 result.

## 12. Limitations

- The study covers four large U.S. equities and five annual development test folds.
- Daily observations and overlapping regime cells are serially dependent; no dependence-aware confidence intervals or significance tests were performed.
- The `n ≥ 30` rule suppresses 32/120 asset/fold and 8/30 macro model-contrast slots, leaving no complete five-fold regime summary.
- LSTM evidence concerns one preregistered architecture/search protocol and three fixed seeds; it does not characterize all recurrent models.
- Fold-local winners differ, so results describe a selection protocol rather than one unchanged fitted model carried across years.
- Runtime measurements come from one local CPU environment and are not a general speed benchmark.
- No trading strategy, transaction costs, portfolio construction, risk analysis, or profitability evaluation was performed.
- PatchTST has no authorized full-benchmark result.
- MAE Skill is baseline-relative MAE, not MASE.

## 13. F1/2025 execution and verification disclosure

The original one-time F1/2025 execution completed prediction generation and sealing, but strict exact-identity verification between sealed and canonical learned-prediction vectors failed after CSV serialization. The frozen protocol therefore recorded **`INVALID_STOP_NO_RERUN`**, and no second execution occurred. A subsequent read-only forensic audit identified a deterministic CSV parse-and-reserialize mechanism that altered last-place floating-point representations and reproduced the saved overall metrics, but it did not convert the run to PASS. Saved 2025 regime outputs also retain additional provenance limitations, including the absence of an end-to-end source-history threshold reconstruction. Consequently, 2025 is excluded from the primary scientific conclusions and does not confirm the D1–D5 findings.

See the [final forensic audit](docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md) and the [public F1 disclosure](docs/F1_INVALID_DISCLOSURE.md). No 2025 performance number or figure is presented in this README, and the underlying F1 files are not distributed in this candidate.

## 14. Reproducibility

### What is available

- Selected content-addressed D1–D5 aggregate metrics, model-selection tables, verification manifests, and run provenance under [`results/`](results).
- Frozen development configurations and contracts under [`configs/`](configs) and [`docs/`](docs).
- Raw and processed data manifests under [`data/`](data), without the provider-sourced market rows.
- Independent saved-artifact checks, including a maximum development-metric reconstruction difference of `1.4710455 × 10⁻¹⁵` against a `10⁻¹²` tolerance and exact (`0.0`) reconstruction for the 16 primary regime artifacts.
- Deterministic figure generation from four saved development CSVs only:

```bash
python docs/figures/generate_figures.py
```

The renderer uses only the Python standard library, verifies each input's content-addressed filename, checks schemas and row counts, and does not train models or run inference.

### Compatible environment

The project metadata requires Python 3.10 or newer. A compatible—not historically bit-locked—installation is:

```bash
python -m pip install -e ".[test,lightgbm,lstm]"
```

The reference runs used Python 3.12.14 on Windows AMD64/CPU, including NumPy 2.5.3, pandas 3.0.5, PyYAML 6.0.3, scikit-learn 1.9.1, LightGBM 4.7.0, SciPy 1.18.1, and PyTorch 2.14.0+cpu. The public candidate omits the original LightGBM and LSTM environment records because they contain local-machine paths. The path-free [baseline run manifest](results/baselines/run_manifest_9927474b9a3ef014ad4e016c48a7f8cda461a0ccb3e0daaab6f124135ddc32d0.json) is retained. These versions describe the historical environment; they do not form a lockfile.

### Safe development checks

The following focused tests use synthetic or fake-provider data and do not fit learned models or access the real snapshot:

```bash
python -m pytest -q tests/test_data_schema.py tests/test_chronology.py tests/test_leakage.py tests/test_walk_forward.py
```

The reusable D1–D5 baseline entry point is:

```bash
python -m src.evaluation.run_baselines --output-root <empty-directory>
```

It requires the exact raw and processed snapshots identified by the included manifests, but those market-data rows are not redistributed. LightGBM and LSTM have genuine D1–D5 historical entry points in [`src/evaluation/run_lightgbm.py`](src/evaluation/run_lightgbm.py) and [`src/evaluation/run_lstm.py`](src/evaluation/run_lstm.py), but they are retained for inspection rather than offered as portable quickstarts: they enforce historical environment, file-hash, interpreter, and empty-output-root gates. The required row-level data, checkpoints, and full historical output trees are absent from this public candidate.

### Reproducibility boundary

This repository supports inspection of aggregate saved results, regeneration of all four figures, and focused synthetic tests. It does **not** support bitwise fresh-clone end-to-end reproduction, row-level metric reconstruction, or learned-model reruns from the exact historical inputs. There is no lockfile, container, or vendored wheel set. Users must obtain market inputs independently under the provider's terms, and a network re-download is not guaranteed to be byte-identical to the frozen snapshot. The [public reproducibility note](docs/PUBLIC_REPRODUCIBILITY.md) records the exact boundary.

The one-time F1 authorization is consumed. No F1 rerun command is provided or permitted. Do not use the unrestricted test suite as a smoke test: it includes real-snapshot and model-fit paths outside this documentation workflow.

## 15. Repository structure

```text
configs/       Frozen development data, walk-forward, model-search, and regime settings
data/          Content manifests only; market-data rows are not distributed
docs/          Scientific contracts, claims ledger, release boundary, and figures
results/       Selected aggregate D1–D5 metrics, selections, and verification manifests
src/           Development data, feature, model, and evaluation implementation
tests/         Focused synthetic data-contract, chronology, leakage, and fold tests
PHASE_2*.md    Selected development and F1-forensic reports
```

Start with the [publication claims ledger](docs/PUBLICATION_CLAIMS_LEDGER.md) for claim-by-claim evidence, the [Phase 2G report](docs/audit/PHASE_2G_REPORT.md) for the combined D1–D5 model audit, the [Phase 2J report](docs/audit/PHASE_2J_REPORT.md) for regime analysis, and the [forensic audit](docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md) for the F1 invalidation boundary.
