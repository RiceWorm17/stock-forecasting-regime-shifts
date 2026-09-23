# Experiment Specification

## When Stock Forecasting Models Fail: Stress-Testing Robustness Under Market Regime Shifts

### A Leakage-Aware Walk-Forward Evaluation of Classical and Deep Learning Models

| Field | Value |
|---|---|
| Specification status | **FROZEN FOR IMPLEMENTATION** |
| Version | 1.1 |
| Purpose | Single source of truth for the V2 empirical pipeline |
| Scope | Daily next-trading-day forecasts for four U.S. stocks, with SPY as the market reference |
| Primary purpose | Measure out-of-sample model robustness under chronological and market-regime shifts |
| Out of scope | Investment advice, trading profitability, and production trading systems |
| Prior evidence | Private forensic audit retained outside this public candidate; its publication-relevant boundary is summarized in `docs/audit/PHASE_2L_FINAL_FORENSIC_AUDIT_REPORT.md` and `docs/F1_INVALID_DISCLOSURE.md` |

Implementation choices must conform to this frozen specification. Any material future change to the target definition, forecast timing, fold boundaries, final-test policy, feature availability, regime definitions, required model families, or primary evaluation metrics must be documented explicitly as a specification amendment rather than changed silently in code.

## 1. Research question

### Main question

> **When stock forecasting models appear effective under normal evaluation settings, how robust are they when evaluated under realistic chronological shifts and different market conditions?**

The project studies **forecast robustness**, not investment performance. Its purpose is to determine whether apparent model skill survives chronological evaluation, comparison with simple assumptions, and market-condition changes.

### Supporting questions

1. Do LightGBM, LSTM, or—if justified—PatchTST consistently outperform a persistence baseline on unseen future periods?
2. Does model performance remain stable across calendar years and across AAPL, MSFT, GOOGL, and NVDA?
3. How much does each model's error or directional accuracy change during high-volatility, negative-trend, or transition periods?
4. Are any gains repeatable across folds and assets, or are they concentrated in a small number of periods?

### Interpretation limits

This study will not claim that:

- stock-market prediction is solved;
- a forecast can be converted directly into a profitable investment strategy;
- higher predictive accuracy implies financial profitability;
- historical robustness guarantees future performance; or
- a complex model is useful merely because its plotted predictions resemble observed prices.

No transaction costs, position sizing, execution assumptions, risk limits, or portfolio rules are modeled. Results must be described as statistical forecasting evidence only.

## 2. Project scope

### Locked initial universe

| Role | Assets |
|---|---|
| Primary prediction targets | AAPL, MSFT, GOOGL, NVDA |
| Market reference | SPY |

SPY is not a fifth headline prediction target in the initial study. It supplies a common market reference for causal market-state definitions and may supply lagged market covariates. Explicit regime labels will not initially be supplied to forecasting models.

### Locked temporal scope

- **Frequency:** Daily U.S. exchange trading sessions
- **Forecast horizon:** The next trading day
- **Initial history:** 2015-01-01 through 2025-12-31, bounded by each asset's actual trading dates
- **Development evaluation:** Calendar-year test periods from 2020 through 2024
- **Untouched final test:** Calendar year 2025
- **Partial 2026 data:** Excluded from V2 version 1.1

The period is deliberately fixed to complete calendar years. It provides multiple market environments while avoiding a moving endpoint and an incomplete final year.

### Forecast timing convention

For a forecast made for asset \(i\) at the close of trading day \(t\):

- all features may use information available by the official close of day \(t\);
- the target is the close-to-close movement from day \(t\) to the asset's next observed trading day \(t+1\);
- no value timestamped after day \(t\) may enter the feature row; and
- predictions are keyed by both `origin_date = t` and `target_date = t+1`.

The project will not forecast weekends or exchange holidays. “Next day” always means the next trading session for that asset.

### Why the scope is controlled

Restricting the study to four large stocks and one market reference improves reliability by:

- allowing every asset/date alignment to be audited;
- keeping the walk-forward and regime matrices interpretable;
- providing enough repeated conditions without creating hundreds of correlated tests;
- making identical preprocessing and evaluation practical for every model;
- limiting model-search and compute costs; and
- leaving time for leakage tests, descriptive variability analysis, and documentation.

Expanding to hundreds of stocks would increase engineering and multiple-comparison risk without answering the core robustness question more clearly.

## 3. Prediction targets

### Price definition

Let \(P_{i,t}\) be the documented split- and dividend-adjusted closing price for asset \(i\) on trading day \(t\). The chosen provider field and adjustment behavior must be recorded in the data manifest before feature generation. Raw and adjusted price fields must not be mixed silently.

### Primary target: next-day log return

The primary regression target is:

\[
r_{i,t+1} = \log\left(\frac{P_{i,t+1}}{P_{i,t}}\right)
\]

Every regression model must output \(\hat r_{i,t+1}\), a forecast of this value. Metrics are calculated in log-return units. A price reconstruction may be shown only as an explanatory visualization:

\[
\hat P_{i,t+1} = P_{i,t}\exp(\hat r_{i,t+1})
\]

Reconstructed prices are not the primary scoring target.

### Secondary target: next-day direction

The secondary binary target is:

\[
y_{i,t+1} =
\begin{cases}
1, & r_{i,t+1} > 0 \\
0, & r_{i,t+1} \le 0
\end{cases}
\]

In the core experiment, a model's directional prediction is derived from the sign of its continuous forecast:

\[
\hat y_{i,t+1} = \mathbb{1}[\hat r_{i,t+1} > 0]
\]

This keeps the model comparison compact and ensures that regression and direction results describe the same forecast. Separate binary classifiers are outside the required V2 version 1.1 experiment and may be proposed later only through change control.

### Why raw price is not the main target

Raw equity price levels are strongly persistent: tomorrow's price is usually close to today's price. A model can therefore obtain a visually convincing price curve, a small-looking level error, or a high \(R^2\) largely by reproducing the current price and long-term trend. The Phase 1 audit demonstrated this problem directly: previous-close persistence outperformed the best submitted LSTM result on matched AAPL, GOOGL, and MSFT holdouts.

Next-day log returns are more appropriate for V2 because they:

- remove much of the mechanically persistent price level;
- make the no-change baseline exactly interpretable as a zero-return forecast;
- provide a comparable target across differently priced stocks;
- make forecast direction explicit; and
- expose whether a model predicts incremental movement rather than copying the latest level.

Return prediction remains difficult and noisy. A near-zero or negative result is scientifically valid.

## 4. Data and leakage rules

### 4.1 Information boundary

For a sample with origin date \(t\) and target date \(t+1\), the entire feature vector must be a function only of values available no later than \(t\). A feature is forbidden if its historical value would change after data from \(t+1\) or later is revealed, except for a documented upstream data correction handled through dataset versioning.

Initial V2 features are restricted to causal transformations of daily market data—OHLCV for the target asset and causally aligned SPY data. Point-in-time fundamentals, news, analyst data, and macro releases are excluded from version 1.1 because their publication and revision timestamps require a separate data contract.

### 4.2 Fold-local fitting and selection

Hyperparameter selection is causal and fold-local. A candidate search space may be specified globally in advance, but the winning configuration for each fold must be selected independently using only that fold's training data and validation data.

At the selection stage, the following are fit using the fold's **training partition only**:

- scalers and normalizers;
- imputation statistics;
- winsorization or clipping thresholds;
- any pre-specified data-driven feature selector;
- learned dimensionality reduction;
- LightGBM, LSTM, and conditional PatchTST model parameters; and
- all other fitted preprocessing state.

The resulting candidates may be evaluated on that fold's validation partition to select hyperparameters, epoch count or early-stopping state, and the winning candidate under the pre-specified selection rule. Test observations and test metrics are never selection inputs.

After selection, the winning preprocessing and model may be refit on that fold's combined training-plus-validation history before producing the fold's test predictions. This refit uses the already selected configuration and fixed training rule; the test period remains untouched.

The causal rule applies separately to every fold:

- D1 selection may use only 2015–2018 training data and 2019 validation data for its 2020 predictions.
- D2 selection may use only 2015–2019 training data and 2020 validation data for its 2021 predictions.
- The same pattern continues through D5 and F1.

No validation or test information from D2 or later may affect D1 selection or predictions. More generally, later folds and future calendar periods must never influence earlier-fold predictions. A prior fold's test score is not a hyperparameter-selection input for a later fold; each fold's winner is determined from that fold's own validation scores.

Required model families and permitted source-feature availability are fixed by this specification. Exact engineered feature definitions, candidate hyperparameter spaces, and selection rules must be recorded before their model family's first fold is generated. Architecture families, allowed feature definitions, candidate search spaces, and selection rules must all be fixed before F1 final-test access.

### 4.3 Forbidden operations

The following are prohibited:

- fitting any scaler, imputer, clipping rule, feature selector, or model on the full dataset before splitting;
- using future prices, future returns, centered rolling windows, or backward-filled future observations;
- random train/test splitting or shuffled cross-validation;
- calculating regime thresholds from validation or test distributions before evaluation;
- labeling day \(t\) with a regime that requires observations after \(t\);
- using financial-statement values before their actual publication timestamp;
- choosing features, hyperparameters, architectures, or plots after inspecting final-test outcomes;
- using any test result, later-fold validation result, or future-period result to select an earlier fold's configuration;
- replacing an earlier fold's already-reported predictions after viewing aggregate development performance;
- using target-period statistics to normalize a target asset or SPY;
- constructing regime-filtered sequences by deleting intervening dates; and
- aligning features, targets, predictions, or assets by row position when a date key is available.

### 4.4 Required split-aware behavior

- All rows must have explicit `asset`, `origin_date`, and `target_date` keys.
- Partition membership is determined by **target date**.
- Rolling features are calculated on the intact chronological series, then samples are assigned to folds.
- Sequence models may carry historical context across a fold boundary, because those prior observations were available at the origin date; targets from the new partition may not enter fitting.
- The first validation/test prediction must use available pre-period context rather than discarding the first 60 or more trading days.
- Validation and test predictions must be joined to targets by `(asset, target_date)`, never by tail length or assumed row offset.
- Missing target returns are never imputed. Rows without a valid next-trading-day target are omitted.
- Missing input handling must be causal. Statistical imputation values must be fit within the fold's training period.
- The implementation must save the fitted transformation state used for every fold/model run.

### 4.5 Required leakage tests

Before learned-model training is permitted, automated tests must show that:

1. perturbing rows after date \(t\) cannot change a feature at or before \(t\);
2. each target equals the independently recomputed next-session log return;
3. each prediction and target share the correct asset and target date;
4. validation/test transforms use objects fitted only on permitted historical rows;
5. regime labels and thresholds use no future or test-period information;
6. sequence endpoints map to the correct next-day targets; and
7. shuffling or reversing source-file enumeration cannot change chronological order.

Financial time series require chronological evaluation because future market distributions, volatility, corporate events, and price extrema are unavailable at the historical forecast origin. Random splitting allows later conditions to influence earlier predictions and produces an unrealistically easy estimate.

## 5. Expanding-window evaluation

### 5.1 Locked fold schedule

All core models use the same annual expanding-window schedule:

| Fold | Training period | Validation period | Test period | Role |
|---|---|---|---|---|
| D1 | 2015–2018 | 2019 | 2020 | Development walk-forward |
| D2 | 2015–2019 | 2020 | 2021 | Development walk-forward |
| D3 | 2015–2020 | 2021 | 2022 | Development walk-forward |
| D4 | 2015–2021 | 2022 | 2023 | Development walk-forward |
| D5 | 2015–2022 | 2023 | 2024 | Development walk-forward |
| F1 | 2015–2023 | 2024 | 2025 | Untouched final evaluation |

Calendar labels refer to the target date. For example, the first 2020 target may use feature history ending on the final 2019 trading session.

### 5.2 Per-fold procedure

For each model family and fold, the complete selection process is rerun causally. Candidate definitions may be global, but candidate scores and the winning configuration are fold-local. For each asset and model family:

1. Construct causal features and targets with explicit timestamps.
2. Fit preprocessing and candidate models on the training period only.
3. Use the validation year for bounded hyperparameter selection and early stopping.
4. Freeze the selected configuration.
5. Refit its preprocessing and model on combined training-plus-validation data using the already selected configuration and a fixed training rule.
6. Generate predictions for the test year with model parameters frozen for that entire test year.
7. At each test origin date, permit only newly observed inputs through that date; do not update model parameters inside the test year.
8. Save row-level predictions and per-asset/per-fold metrics using the canonical keys.

The fixed-within-year policy deliberately tests robustness as conditions move away from the most recent fit. A future study may compare more frequent retraining, but it is outside V2 version 1.1.

### 5.3 Fold-local selection and retrospective aggregation

Within each fold, the winning configuration is selected from that fold's candidate validation results only. The primary selection statistic is that fold's macro-averaged validation MAE across the four stocks, with each stock weighted equally. RMSE, directional accuracy, runtime, and parameter count from the same training/validation history are secondary evidence. If two candidates have practically indistinguishable validation MAE, select the simpler or less computationally expensive candidate.

D1–D5 test metrics may be aggregated only after every contributing prediction has been generated under its own causal, fold-local rule. These aggregate statistics may be used for:

- retrospective development analysis;
- summarizing chronological and regime robustness; and
- deciding whether the optional PatchTST family is worth evaluating further, provided 2025 remains untouched.

Aggregate D1–D5 results must not be used to retroactively alter, replace, or retune already-reported D1–D5 predictions. If aggregate evidence triggers the optional PatchTST gate, PatchTST's candidate space must be fixed before its first run, and its configuration in each historical fold must still be selected only from that fold's training and validation data. The family-level decision to investigate PatchTST is reported as conditional exploratory development evidence; it does not change required-model predictions.

### 5.4 Why random 80/20 splitting is prohibited

A random split mixes earlier and later market states, allows neighboring observations from the same episode into both sets, and tests interpolation rather than forecasting into the future. The expanding schedule preserves the actual information order, exposes year-to-year degradation, and gives every model the same historical opportunity set.

## 6. Final test policy

### Validation versus final test

| Period | Permitted use |
|---|---|
| Validation | Within its own fold, choose among preregistered configurations, set bounded hyperparameters, determine early-stopping or epoch rules, and diagnose that fold's development behavior |
| Final test | One-time estimate of the already frozen pipeline's out-of-sample performance; no design or selection decisions |

Validation is part of model development. It may be inspected and used to make choices. The final test simulates data that did not exist during development and must remain untouched until the code, model families, feature contract, regime rules, metrics, and output schema are frozen.

### Locked final procedure

1. Complete D1–D5 and freeze the architecture families, permitted feature definitions, candidate hyperparameter spaces, fold-local selection rules, regime definitions, and metrics for version 1.1.
2. Record code revision, environment lock, data hashes, configurations, and random seeds.
3. Select the F1 winning configuration using only 2015–2023 training data and 2024 validation data under the already fixed candidate space and selection rule. D1–D5 aggregate test statistics cannot select the F1 winner.
4. Freeze the chosen F1 configuration and training duration.
5. Refit the selected pipeline on 2015–2024 without accessing 2025 targets.
6. Evaluate 2025 exactly once for the primary final report.
7. Preserve row-level predictions and an immutable evaluation manifest.

The 2025 final test must not influence:

- feature creation or removal;
- normalization choices;
- model-family inclusion;
- hyperparameter or architecture decisions;
- random-seed selection;
- regime definitions or thresholds;
- metric definitions; or
- which results and plots are reported.

If a genuine implementation defect is discovered after final-test evaluation, the run is marked invalid. A fix must be justified independently of performance, the specification/code version must be incremented, and the complete affected benchmark must be rerun and disclosed. The final test may not be repeatedly queried to improve results.

## 7. Market-regime analysis

### 7.1 Role of regime

In V2 version 1.1, regime is **an evaluation dimension, not an explicit model input or routing rule**. Models are trained on intact chronological data. Regime labels are joined to predictions afterward to measure robustness.

Lagged SPY market features may be used as ordinary causal covariates, but the categorical regime label itself must not be passed to the model. No separate bull, bear, or neutral model is trained initially.

### 7.2 Regime source and timestamp

Initial market regimes are derived from SPY so that all four stocks share the same market-wide label on a date. Every label attached to a forecast for \(t+1\) is computed at origin date \(t\), using SPY observations no later than \(t\).

The following backward-looking quantities are used:

\[
Trend63_t = \log\left(\frac{P^{SPY}_t}{P^{SPY}_{t-63}}\right)
\]

\[
RV21_t = \operatorname{StdDev}(r^{SPY}_{t-20},\ldots,r^{SPY}_t)\sqrt{252}
\]

Thresholds based on distributions are fit without the evaluation period:

- validation labels use thresholds estimated from that fold's training period;
- test labels use thresholds estimated from combined training-plus-validation history; and
- no threshold is recomputed from the test year's distribution.

### 7.3 Locked regime dimensions

Regime dimensions are overlapping evaluation labels, not mutually exclusive model datasets.

| Dimension | Category | Causal definition at origin date \(t\) |
|---|---|---|
| Trend | Positive trend | `Trend63_t > 0` |
| Trend | Negative trend | `Trend63_t <= 0` |
| Volatility | Low volatility | `RV21_t <= median(RV21)` in the permitted fitting history |
| Volatility | High volatility | `RV21_t > median(RV21)` in the permitted fitting history |
| Transition | Large volatility increase | `RV21_t / RV21_(t-21) >= 1.5` |
| Transition | Abnormal return | `abs(SPY return_t)` exceeds the 95th percentile of absolute SPY returns in the permitted fitting history |
| Transition | Shifted | Large volatility increase **or** abnormal return |
| Transition | Stable/normal | Neither transition condition is true |

A zero or unavailable denominator prevents the volatility-ratio flag rather than producing an infinite transition. Rows lacking the required 63-day or 21-day history are excluded from the corresponding regime analysis but may remain in model evaluation if their model features are otherwise valid.

### 7.4 Regime reporting rules

- Report trend, volatility, and transition contrasts separately.
- Do not delete intervening dates before constructing model sequences.
- Do not hand-label famous crises or rallies using later knowledge.
- Report sample count with every regime metric.
- A per-asset/per-fold regime cell with fewer than 30 target observations is descriptive only and cannot support a headline comparison.
- Aggregate results must give each target stock equal weight; large assets or long cells may not dominate solely through row count.
- The primary “normal versus shifted” contrast is stable/normal versus transition/shifted. High-versus-low volatility and negative-versus-positive trend are separate robustness views.

## 8. Model comparison

### 8.1 Persistence baseline — required first

For regression, price persistence assumes:

\[
\hat P_{i,t+1}=P_{i,t}
\quad\Longleftrightarrow\quad
\hat r_{i,t+1}=0
\]

This is the primary baseline and the minimum hurdle for every learned model. It has no fitted parameters.

For the secondary direction view, also report direction persistence:

\[
\hat y_{i,t+1}=\mathbb{1}[r_{i,t}>0]
\]

This prevents directional accuracy from being interpreted without a simple directional reference. Baseline results must be produced before any learned model is trained.

### 8.2 LightGBM — required classical model

Purpose: provide a strong tabular reference for causal lag and rolling features.

Requirements:

- one regression model per target asset unless a pooled design is approved before implementation;
- the same approved feature families and fold schedule across assets;
- bounded fold-local hyperparameter selection using training and validation only;
- deterministic seeds and saved configuration;
- no test-period feature selection; and
- return forecasts and sign-derived directional predictions in the canonical result schema.

LightGBM is the principal learned benchmark because it can model nonlinear feature interactions without the training cost and data appetite of a deep sequence model.

### 8.3 LSTM — required deep-learning model

Purpose: provide continuity with the original project while testing whether a corrected recurrent sequence model adds value.

Requirements:

- a small architecture with a preregistered context-length grid;
- correctly aligned sequence windows carrying context into validation/test periods;
- fold-local preprocessing and saved scalers;
- early stopping using validation only;
- at least three fixed random seeds for reported development results;
- identical fold dates and output keys to the other models; and
- no architecture expansion after final-test access.

The LSTM is included because it is relevant to the original work, not because a deep model is assumed to win.

### 8.4 PatchTST — conditional modern time-series model

Purpose: test whether a patch-based transformer offers stable value from longer temporal context.

PatchTST is included only if all of the following gates are satisfied:

1. The data pipeline, leakage tests, persistence baseline, LightGBM, and LSTM are complete and reproducible.
2. The earliest development fold has enough valid pooled training windows for the chosen context without aggressive duplication or synthetic sampling.
3. A pilot on D1 fits available memory and completes within a documented, reasonable local compute budget.
4. The implementation uses the same targets, folds, timestamps, regime analysis, and artifact schema as other models.
5. The code can be maintained without a one-off pipeline or hidden preprocessing path.
6. The tuning budget is fixed before the pilot and does not expand in response to weak results.

PatchTST is not added merely because it is newer. A horizon-one daily task with four target stocks may be too small to justify a transformer. If a bounded pilot is unstable or fails to show credible validation value, the project will document that result and omit full PatchTST evaluation.

### 8.5 Model-search limits

Controlled evaluation takes priority over broad search:

- Baselines have no tuning.
- LightGBM and LSTM receive small, documented candidate grids fixed before their first fold is generated. The grid may be global, but the winning candidate is selected independently within every fold.
- Neural results use the same fixed set of at least three seeds; seeds may not be dropped because of poor outcomes.
- PatchTST, if admitted, receives no more candidate configurations or seeds than the LSTM without a documented specification revision.
- Each added configuration must answer a stated methodological question; blind architecture search is prohibited.

## 9. Metrics and robustness measures

### 9.1 Required regression metrics

For \(N\) next-day log-return predictions:

\[
MAE = \frac{1}{N}\sum_{j=1}^{N}|r_j-\hat r_j|
\]

\[
RMSE = \sqrt{\frac{1}{N}\sum_{j=1}^{N}(r_j-\hat r_j)^2}
\]

MAE is the primary model-selection metric. RMSE is required because it places more weight on large misses during unstable periods.

### 9.2 Required directional metric

\[
Directional\ Accuracy = \frac{1}{N}\sum_{j=1}^{N}\mathbb{1}[y_j=\hat y_j]
\]

Actual or predicted returns equal to zero map to class 0 under the target definition. Directional accuracy must be shown beside class balance and the direction-persistence baseline.

### 9.3 Baseline-relative skill

Absolute errors naturally increase when return magnitudes and volatility increase. Raw error degradation alone therefore cannot distinguish “the market became intrinsically harder for every forecaster” from “the learned model lost skill relative to a simple assumption.” Every regime comparison must include a baseline-relative measure evaluated on the exact same observations.

The primary baseline-relative metric is MAE skill against the pre-specified zero-return persistence baseline:

\[
MAE\ Skill = 1-\frac{MAE_{model}}{MAE_{baseline}}
\]

The model and baseline MAE must use identical asset/date rows within the fold and regime cell. If baseline MAE is zero, skill is undefined and must be reported as such rather than forced to a numeric value.

Interpretation:

- `MAE Skill > 0`: the model beats persistence;
- `MAE Skill = 0`: the model matches persistence; and
- `MAE Skill < 0`: the model performs worse than persistence.

RMSE-relative skill may be reported as a secondary measure using the same formulation, but MAE skill remains the primary simple relative-robustness measure unless a future specification amendment justifies a change.

### 9.4 Raw and baseline-relative degradation

For any raw metric \(m\), the required shift contrast remains:

\[
\Delta_m = m_{shifted}-m_{normal}
\]

Interpretation:

- for MAE and RMSE, positive \(\Delta\) means larger absolute error under shift;
- for directional accuracy, negative \(\Delta\) means worse direction accuracy under shift.

For a common “larger means less robust” directional view, also report:

\[
Loss_{DA}=DA_{normal}-DA_{shifted}
\]

The required regime-relative MAE skill degradation is:

\[
Skill\ Degradation = MAE\ Skill_{shifted}-MAE\ Skill_{normal}
\]

A negative value means the model's advantage relative to persistence deteriorated in the shifted regime. A non-negative value means relative skill was maintained or improved, even if raw errors rose for both model and baseline. This statistic describes an association within the evaluation design; it does not by itself establish that the regime causally made the model fail.

The same raw difference convention is used for high-minus-low volatility and negative-minus-positive trend comparisons. Both **raw error degradation** and **baseline-relative MAE skill degradation** are mandatory; neither may substitute for the other.

### 9.5 Aggregation and uncertainty scope

Core reporting requires:

- metrics per model, asset, fold, and test period;
- metrics per model and regime category with sample counts;
- a macro-average giving each of AAPL, MSFT, GOOGL, and NVDA equal weight;
- each D1–D5 fold value; and
- descriptive fold-to-fold variability.

The default development-fold summary is the median across D1–D5 with the interquartile range. Mean and standard deviation may be shown as supplemental descriptive summaries. All such quantities must be labeled **descriptive variability**, not confidence intervals or formal inferential uncertainty.

Formal statistical confidence intervals are not required for the core project and must not be added to the initial data-pipeline or model phases. They remain an optional future extension. If later approved through change control, the method must respect serial dependence—for example, a moving-block bootstrap, stationary bootstrap, or another explicitly justified time-series method. An ordinary IID bootstrap across daily observations must not be used without a defensible dependence argument.

The headline is not merely the highest average score. A model that wins once and fails badly during shifts is less robust than one with slightly weaker average error and stable fold/regime behavior.

An optional MASE value may be added only if its scaling denominator is defined from the permitted training history using the standard naïve-error construction. A custom ratio to the zero-return baseline must be labeled “baseline-relative MAE,” not MASE.

## 10. Experiment matrix

| Experiment | Status | Question | Required comparison | Primary outputs |
|---|---|---|---|---|
| A. Baseline comparison | Required | Do complex models beat simple assumptions? | Persistence vs LightGBM vs LSTM; PatchTST only if gated in | MAE/RMSE difference from persistence, MAE skill, directional accuracy vs direction persistence, wins by asset/fold |
| B. Walk-forward evaluation | Required | Do models maintain performance over time? | Identical D1–D5 and F1 folds for every admitted model | Per-year metrics, macro-average, descriptive fold dispersion, model-ranking stability |
| C. Regime stress test | Required | Which models degrade most during unstable periods? | Normal vs shifted, low vs high volatility, positive vs negative trend | Raw error degradation, MAE skill degradation, directional stability loss, sample counts, descriptive fold dispersion |
| D. Cross-stock validation | Optional | Do conclusions generalize across assets? | Leave-one-stock-out pooled training, if the core study is complete | Held-out-stock metrics and comparison with per-stock models |

### Experiment A — baseline comparison

Run persistence first on the exact target rows intended for learned models. Report MAE skill on those same rows. A learned model provides evidence of added forecasting value only if its improvement is repeated across more than one development fold and is not driven by one stock. The final report must show losses as well as wins.

### Experiment B — walk-forward evaluation

Run the locked expanding folds without random splits. Compare both average performance and how rankings change from 2020 through 2024. After all structural choices and the causally selected F1 winner are frozen, run F1 once on 2025.

### Experiment C — regime stress test

Join the causal SPY-derived regime labels to the unchanged row-level forecast table. Evaluate both raw degradation and baseline-relative skill degradation without training separate regime models or removing dates from sequences. The central result is stability under shifts, not only normal-period accuracy.

### Experiment D — optional cross-stock validation

This experiment may begin only after A–C are complete. If run, train a pooled model on three stocks and evaluate the fourth as a held-out asset under the same chronological boundaries, rotating through all four. SPY remains a reference, not a held-out target. Scaling and asset encodings must not learn from the held-out asset's test period.

Cross-stock validation is optional because it materially increases design and compute complexity. Omitting it does not make the core project incomplete.

## 11. Explicit exclusions

V2 version 1.1 excludes:

- real or paper trading strategies;
- portfolio construction or optimization;
- transaction-cost, slippage, or execution simulation;
- reinforcement learning;
- live data ingestion or live trading;
- broker or exchange integration;
- financial advice or investment recommendations;
- hundreds of stocks or unconstrained cross-sectional expansion;
- intraday, tick, cryptocurrency, option, or order-book data;
- point-in-time fundamentals, news, sentiment, and macroeconomic releases;
- dozens of model architectures;
- exhaustive or adaptive model search against test outcomes;
- regime-specific model routing in the initial experiment; and
- profitability claims inferred from statistical accuracy.

These exclusions are deliberate. The research contribution is a controlled, leakage-aware robustness evaluation. More data sources and architectures would add ways to overfit, leak, or obscure the core comparison before its foundations are validated.

## 12. Implementation roadmap

The phases below are implementation stages inside V2; they do not alter the completed Phase 1 audit.

### Phase 1 — Data pipeline

Deliverables:

- immutable raw daily data for AAPL, MSFT, GOOGL, NVDA, and SPY;
- fixed provider/date/adjustment manifest and file hashes;
- validated asset/calendar schema;
- target construction with origin and target dates;
- causal feature pipeline;
- expanding-fold registry; and
- leakage, chronology, and alignment tests.

**Exit gate:** every required data/leakage test passes, and a reviewer can trace any sample from raw values to target and fold.

### Phase 2 — Baseline models

Deliverables:

- zero-return price-persistence predictions;
- previous-direction persistence predictions;
- canonical row-level prediction table;
- MAE, RMSE, directional accuracy, and baseline comparison tables;
- baseline-relative MAE skill on exactly matched observations; and
- a reproducible one-command baseline run.

**Exit gate:** baselines run across D1–D5 with exact date alignment before any learned model training.

### Phase 3 — LightGBM and LSTM

Deliverables:

- bounded, preregistered LightGBM and LSTM configurations;
- fold-local fitted artifacts and manifests;
- multi-seed LSTM results;
- identical walk-forward outputs for both families; and
- comparisons against persistence by asset and fold.

**Exit gate:** all outputs conform to one schema and can be regenerated from locked inputs and configuration.

### Phase 4 — PatchTST evaluation if justified

Deliverables:

- written gate decision;
- bounded D1 pilot if admitted;
- compute/memory record;
- full identical-fold results only if the pilot passes; or
- a documented omission/negative result if it does not.

**Exit gate:** PatchTST adds no separate preprocessing/evaluation path and does not delay completion of the core study.

### Phase 5 — Regime analysis

Deliverables:

- causal SPY trend, volatility, and transition labels;
- sample-size audits;
- normal-versus-shifted degradation tables;
- high-versus-low volatility and negative-versus-positive trend contrasts;
- raw-error and baseline-relative-skill degradation;
- per-asset and macro results with descriptive fold-to-fold dispersion; and
- final F1 evaluation after all structural decisions are frozen and the F1 winner is selected only from its permitted training/validation history.

**Exit gate:** every robustness claim is linked to a preregistered contrast, sample count, and underlying row-level predictions.

### Phase 6 — Documentation and GitHub polishing

Deliverables:

- root README with research question, non-financial-use disclaimer, method, and reproducible commands;
- locked dependency/environment files;
- data card and model cards;
- experiment and artifact manifests;
- concise consolidated tables and legible figures;
- limitations and negative-result discussion; and
- archived V1 provenance without presenting invalid V1 outputs as V2 evidence.

**Exit gate:** a fresh environment can reproduce the documented results without editing source constants or absolute paths.

## 13. Governance and acceptance criteria

### Canonical artifact requirements

Every prediction row must contain at least:

`run_id`, `spec_version`, `data_version`, `model`, `model_config_id`, `seed`, `fold`, `asset`, `origin_date`, `target_date`, `actual_log_return`, `predicted_log_return`, `actual_direction`, `predicted_direction`, and the three regime labels.

Every run must record:

- source revision;
- environment lock/hash;
- raw and processed data hashes;
- fold definition;
- fitted preprocessing artifact identifiers;
- model configuration and seed;
- training/validation date bounds; and
- runtime status and exceptions.

### Core completion criteria

V2 version 1.1 is complete when:

1. the data and leakage tests pass;
2. persistence, LightGBM, and LSTM have comparable D1–D5 results;
3. the PatchTST gate has a documented pass/omit decision;
4. architecture families, feature definitions, candidate spaces, selection rules, metrics, and regime rules are frozen before F1 final-test access, with only the permitted F1 fold-local selection remaining;
5. the 2025 final test is evaluated once under the stated policy;
6. Experiments A–C are reported with row counts, raw and baseline-relative degradation, and descriptive fold-to-fold dispersion;
7. failures to beat persistence are reported plainly; and
8. all headline tables can be reproduced from saved row-level predictions.

No minimum accuracy or profitability threshold is required for project success. A well-supported finding that sophisticated models are unstable or do not beat persistence satisfies the research objective.

### Change control

Before 2025 final-test access, a material change is allowed only if this document is versioned and the change log records the rationale and affected experiments. This requirement specifically covers the target definition, forecast timing, fold boundaries, final-test policy, feature availability, regime definitions, required model families, and primary evaluation metrics. Such changes must be specification amendments, never silent code edits. After final-test access, performance-motivated changes are prohibited; a corrected study requires a new specification version and full disclosure.

| Version | Date | Change | Final test accessed? |
|---|---|---|---|
| 1.0 | 2026-09-13 | Initial experiment contract created from the completed Phase 1 audit | No |
| 1.1 | 2026-09-13 | Clarified fold-local causal selection, added baseline-relative robustness, froze uncertainty scope, and declared the contract frozen for implementation | No |

## Recommended next phase

**Phase 2B:** Implement the immutable data contract, target-generation logic, walk-forward fold registry, and automated leakage/alignment tests before any predictive modeling. This corresponds to implementation Phase 1 in the roadmap above. Do not train LightGBM, LSTM, or PatchTST until those tests pass and the persistence baseline can run end to end.
